---
title: Automating Trading Strategies in MQL5 (Part 29): Creating a price action Gartley Harmonic Pattern system
url: https://www.mql5.com/en/articles/19111
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 10
scraped_at: 2026-01-22T17:18:27.879160
---

[![](https://www.mql5.com/ff/sh/0hvxp984jjj79943z2/6373d9e5710a718ffa6a7d50a5db9dd1.jpg)\\
Web terminal on your iPhone or Android\\
\\
Full-featured MetaTrader 5 platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=uyigsjnbfcdvysiynusmriwvhincciwd&s=c95531ae2fd8a81b0fac3def2e4cf820a67584bbf4b02f76ec75f808942dbbd2&uid=&ref=https://www.mql5.com/en/articles/19111&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049034017312973745)

MetaTrader 5 / Trading


### Introduction

In our [previous article (Part 28)](https://www.mql5.com/en/articles/19105), we developed a Bat Pattern system in [MetaQuotes Language 5](https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5 "https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5") (MQL5) that detected bullish and bearish [Bat harmonic patterns](https://www.mql5.com/go?link=https://howtotrade.com/chart-patterns/bat-pattern/ "https://howtotrade.com/chart-patterns/bat-pattern/") using precise Fibonacci ratios. In Part 29, we create a [Gartley Pattern](https://www.mql5.com/go?link=https://www.investopedia.com/terms/g/gartley.asp "https://www.investopedia.com/terms/g/gartley.asp") program that identifies bullish and bearish Gartley harmonic patterns through pivot points and specific Fibonacci retracements, executing trades with dynamic entry and multi-level take-profit points, enhanced by visual triangles, trendlines, and labels for clear pattern representation. We will cover the following topics:

1. [Understanding the Gartley Harmonic Pattern Framework](https://www.mql5.com/en/articles/19111#para1)
2. [Implementation in MQL5](https://www.mql5.com/en/articles/19111#para2)
3. [Backtesting](https://www.mql5.com/en/articles/19111#para3)
4. [Conclusion](https://www.mql5.com/en/articles/19111#para4)

By the end, you’ll have a powerful MQL5 strategy for Gartley harmonic pattern trading, ready for customization—let’s dive in!

### Understanding the Gartley Harmonic Pattern Framework

The [Gartley pattern](https://www.mql5.com/go?link=https://www.investopedia.com/terms/g/gartley.asp "https://www.investopedia.com/terms/g/gartley.asp") is a harmonic trading formation defined by five key swing points—X, A, B, C, and D—and exists in two forms: a bullish pattern and a bearish pattern, each designed to identify high-probability reversal zones using specific [Fibonacci ratios](https://www.metatrader5.com/en/terminal/help/objects/fibo/fibo_retracement "https://www.metatrader5.com/en/terminal/help/objects/fibo/fibo_retracement"). In a bullish Gartley, the structure forms a low-high-low-high-low sequence where point X is a swing low, point A a swing high, point B a swing low (retracing approximately 0.618 of XA), point C a swing high (extending 0.382 to 0.886 of AB), and point D a swing low (retracing 0.786 of XA, positioned above X). Conversely, a bearish Gartley forms a high-low-high-low-high sequence, with point X as a swing high, point A a swing low, point B a swing high, point C a swing low, and point D a swing high (retracing 0.786 of XA, positioned below X). Below are the visualized pattern types.

Bullish Gartley Harmonic Pattern:

![BULLISH GARTLEY HARMONIC PATTERN](https://c.mql5.com/2/162/Screenshot_2025-08-08_172642.png)

Bearish Gartley Harmonic Pattern:

![BEARISH GARTLEY HARMONIC PATTERN](https://c.mql5.com/2/162/Screenshot_2025-08-08_172625.png)

To identify the patterns, below is our structured approach:

- Defining the XA Leg: The initial move from point X to point A establishes the pattern’s direction (upward for bullish, downward for bearish) and serves as the reference for subsequent Fibonacci calculations.
- Establishing the AB Leg: Point B should retrace approximately 0.618 of the XA leg, indicating a significant but controlled correction of the initial impulse.
- Analyzing the BC Leg: This leg should extend between 0.382 and 0.886 of the AB leg, setting up a counter-move that precedes the final reversal zone.
- Setting the CD Leg: The final leg should retrace 0.786 of the XA leg, marking the potential reversal zone at point D, where the pattern completes and a trade signal is generated.

By applying these [geometric and Fibonacci-based criteria](https://www.metatrader5.com/en/terminal/help/objects/fibo/fibo_retracement "https://www.metatrader5.com/en/terminal/help/objects/fibo/fibo_retracement"), our trading system will systematically detect valid Gartley patterns in price data. Once a pattern is confirmed, the system will visualize the formation on the chart with triangles, trend lines, and labels for points X, A, B, C, and D, alongside trade levels for entry and take profits. This setup will enable automated execution of trades at the D point with calculated stop loss and multi-level take-profit zones, leveraging the pattern’s predictive power for market reversals. Let’s proceed to the implementation!

### Implementation in MQL5

To create the program in MQL5, open the [MetaEditor](https://www.metatrader5.com/en/automated-trading/metaeditor "https://www.metatrader5.com/en/automated-trading/metaeditor"), go to the Navigator, locate the Indicators folder, click on the "New" tab, and follow the prompts to create the file. Once it is made, in the coding environment, we will need to declare some [global variables](https://www.mql5.com/en/docs/basis/variables/global) that we will use throughout the program.

```
//+------------------------------------------------------------------+
//|                                       Gartley Pattern EA.mq5     |
//|                        Copyright 2025, Forex Algo-Trader, Allan. |
//|                                 "https://t.me/Forex_Algo_Trader" |
//+------------------------------------------------------------------+
#property copyright "Forex Algo-Trader, Allan"
#property link      "https://t.me/Forex_Algo_Trader"
#property version   "1.00"
#property description "This EA trades based on Gartley Strategy"
#property strict

//--- Include the trading library for order functions
#include <Trade\Trade.mqh>  //--- Include Trade library
CTrade obj_Trade;  //--- Instantiate a obj_Trade object

//--- Input parameters for user configuration
input int    PivotLeft    = 5;      // Number of bars to the left for pivot check
input int    PivotRight   = 5;      // Number of bars to the right for pivot check
input double Tolerance    = 0.10;   // Allowed deviation (10% of XA move)
input double LotSize      = 0.01;   // Lot size for new orders
input bool   AllowTrading = true;   // Enable or disable trading

//---------------------------------------------------------------------------
//--- Gartley pattern definition:
//
//--- Bullish Gartley:
//---   Pivots (X-A-B-C-D): X swing low, A swing high, B swing low, C swing high, D swing low.
//---   Normally XA > 0; Ideal B = A - 0.618*(A-X); Legs within specified ranges; D at 0.786 retracement.
//
//--- Bearish Gartley:
//---   Pivots (X-A-B-C-D): X swing high, A swing low, B swing high, C swing low, D swing high.
//---   Normally XA > 0; Ideal B = A + 0.618*(X-A); Legs within specified ranges; D at 0.786 retracement.
//---------------------------------------------------------------------------

//--- Structure for a pivot point
struct Pivot {
   datetime time;   //--- Bar time of the pivot
   double   price;  //--- Pivot price (High for swing high, low for swing low)
   bool     isHigh; //--- True if swing high; false if swing low
};

//--- Global dynamic array for storing pivots in chronological order
Pivot pivots[];     //--- Declare a dynamic array to hold identified pivot points

//--- Global variables to lock in a pattern (avoid trading on repaint)
int      g_patternFormationBar = -1;  //--- Bar index where the pattern was formed (-1 means none)
datetime g_lockedPatternX      = 0;   //--- The key X pivot time for the locked pattern
```

To establish the foundation for the [Gartley Pattern](https://www.mql5.com/go?link=https://www.investopedia.com/terms/g/gartley.asp "https://www.investopedia.com/terms/g/gartley.asp"), we first include the "<Trade\\Trade.mqh>" library and instantiate "obj\_Trade" as a [CTrade](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade) object to handle trade operations such as executing buy and sell orders. Then, we proceed to define [input parameters](https://www.mql5.com/en/docs/basis/variables/inputvariables) for user customization: "PivotLeft" and "PivotRight" at 5 bars each to specify the lookback range for pivot detection, "Tolerance" at 0.10 to allow a 10% deviation in [Fibonacci](https://www.metatrader5.com/en/terminal/help/objects/fibo/fibo_retracement "https://www.metatrader5.com/en/terminal/help/objects/fibo/fibo_retracement") ratios, "LotSize" at 0.01 for trade volume, and "AllowTrading" as true to enable automated trading.

Next, we define the "Pivot" [structure](https://www.mql5.com/en/docs/basis/types/classes) with "time" (datetime), "price" ( [double](https://www.mql5.com/en/docs/basis/types/double)), and "isHigh" (bool) to store swing points, declare "pivots" as a dynamic array to hold these points, and initialize globals "g\_patternFormationBar" to -1 to track the bar where a pattern forms and "g\_lockedPatternX" to 0 to lock the X pivot time for pattern confirmation, providing the core framework for detecting and trading Gartley patterns. For visualization, we can have functions to draw lines, labels, and triangles.

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

We proceed to implement visualization functions for the system to create clear chart representations of the pattern and its trade levels. First, we develop the "DrawTriangle" function, which uses [ObjectCreate](https://www.mql5.com/en/docs/objects/objectcreate) to draw a filled triangle ( [OBJ\_TRIANGLE](https://www.mql5.com/en/docs/constants/objectconstants/enum_object/obj_triangle)) defined by three points with times ("t1", "t2", "t3") and prices ("p1", "p2", "p3"), setting [OBJPROP\_COLOR](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer) to the specified color, "OBJPROP\_STYLE" to [STYLE\_SOLID](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer), "OBJPROP\_WIDTH" to the given width, "OBJPROP\_FILL" to enable or disable filling, and "OBJPROP\_BACK" to set background or foreground placement using the [ObjectSetInteger](https://www.mql5.com/en/docs/objects/objectsetinteger) function.

Then, we proceed to create the "DrawTrendLine" function, which draws a trend line ( [OBJ\_TREND](https://www.mql5.com/en/docs/constants/objectconstants/enum_object/obj_trend)) between two points. Next, we implement the "DrawDottedLine" function, which creates a horizontal dotted line ("OBJ\_TREND") at a specified price from "t1" to "t2". Last, we develop the "DrawTextEx" function, which creates a text label ( [OBJ\_TEXT](https://www.mql5.com/en/docs/constants/objectconstants/enum_object/obj_text)) at coordinates ("t", "p"), setting "OBJPROP\_TEXT" to the specified text, "OBJPROP\_COLOR", "OBJPROP\_FONTSIZE", and "OBJPROP\_FONT" to "Arial Bold" using [ObjectSetString](https://www.mql5.com/en/docs/objects/objectsetstring) and "ObjectSetInteger", anchoring above for swing highs or below for lows based on "isHigh" with [OBJPROP\_ANCHOR](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer), and centering with "OBJPROP\_ALIGN", ensuring the Gartley pattern and its trade levels are visually clear on the chart. We can now proceed to the [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) event handler and try to find pivot points that we can use later on for pattern identification. Here is the logic we use to achieve that.

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

Here, we implement the initial logic of the [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) function to detect swing pivots, forming the foundation for identifying Gartley harmonic patterns. First, we declare a static "lastBarTime" initialized to 0 to track the last processed bar and compare it with "currentBarTime" obtained from [iTime](https://www.mql5.com/en/docs/series/itime) at shift 1 for the current symbol and period, exiting if unchanged to prevent redundant processing, and updating "lastBarTime" when a new bar is detected. Then, we proceed to clear the "pivots" array with [ArrayResize](https://www.mql5.com/en/docs/array/arrayresize) to ensure a fresh analysis.

Next, we retrieve the total number of bars with [Bars](https://www.mql5.com/en/docs/series/bars), set the pivot detection range with "start" as "PivotLeft" and "end" as total bars minus "PivotRight", and iterate through bars from "end - 1" to "start". For each bar, we assume it’s a swing high ("isPivotHigh" true) and low ("isPivotLow" true), obtain its high and low prices using [iHigh](https://www.mql5.com/en/docs/series/ihigh) and [iLow](https://www.mql5.com/en/docs/series/ilow), and validate the pivot by checking surrounding bars within "PivotLeft" and "PivotRight" with "iHigh" and "iLow", invalidating the pivot if any neighboring bar has a higher high or lower low. Last, if the bar qualifies as a pivot, we create a "Pivot" [structure](https://www.mql5.com/en/docs/basis/types/classes), set its "time" with "iTime", "price" to the high or low based on "isPivotHigh", and "isHigh" flag, then append it to the "pivots" array using [ArrayResize](https://www.mql5.com/en/docs/array/arrayresize) and store it. We get the following array of data when we print the pivot structure.

![PIVOTS DATA STRUCTURE](https://c.mql5.com/2/162/Screenshot_2025-08-08_193140.png)

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

//--- Initialize a flag to indicate if a valid Gartley pattern is found
bool patternFound = false;
//--- Initialize pattern type
string patternType = "";
//--- Check for the high-low-high-low-high (Bearish reversal) structure
if(X.isHigh && (!A.isHigh) && B.isHigh && (!C.isHigh) && D.isHigh) {
   //--- Calculate the difference between pivot X and A
   double diff = X.price - A.price;
   //--- Ensure the difference is positive
   if(diff > 0) {
      //--- Calculate the ideal position for pivot B based on Fibonacci ratio
      double idealB = A.price + 0.618 * diff;
      //--- Check if actual B is within tolerance of the ideal position
      if(MathAbs(B.price - idealB) <= Tolerance * diff) {
         //--- Calculate the AB leg length
         double AB = B.price - A.price;
         //--- Calculate the BC leg length
         double BC = B.price - C.price;
         //--- Verify that BC is within the acceptable Fibonacci range
         if((BC >= 0.382 * AB) && (BC <= 0.886 * AB)) {
            //--- Calculate the retracement
            double retrace = D.price - A.price;
            //--- Verify that the retracement is within tolerance of 0.786 and that D is below X
            if(MathAbs(retrace - 0.786 * diff) <= Tolerance * diff && (D.price < X.price)) {
               patternFound = true;
               patternType = "Bearish";
            }
         }
      }
   }
}
//--- Check for the low-high-low-high-low (Bullish reversal) structure
if((!X.isHigh) && A.isHigh && (!B.isHigh) && C.isHigh && (!D.isHigh)) {
   //--- Calculate the difference between pivot A and X
   double diff = A.price - X.price;
   //--- Ensure the difference is positive
   if(diff > 0) {
      //--- Calculate the ideal position for pivot B based on Fibonacci ratio
      double idealB = A.price - 0.618 * diff;
      //--- Check if actual B is within tolerance of the ideal position
      if(MathAbs(B.price - idealB) <= Tolerance * diff) {
         //--- Calculate the AB leg length
         double AB = A.price - B.price;
         //--- Calculate the BC leg length
         double BC = C.price - B.price;
         //--- Verify that BC is within the acceptable Fibonacci range
         if((BC >= 0.382 * AB) && (BC <= 0.886 * AB)) {
            //--- Calculate the retracement
            double retrace = A.price - D.price;
            //--- Verify that the retracement is within tolerance of 0.786 and that D is above X
            if(MathAbs(retrace - 0.786 * diff) <= Tolerance * diff && (D.price > X.price)) {
               patternFound = true;
               patternType = "Bullish";
            }
         }
      }
   }
}
```

First, we determine the total number of pivots with " [ArraySize(pivots)](https://www.mql5.com/en/docs/array/arraysize)" stored in "pivotCount" and exit if fewer than 5 pivots are found, resetting "g\_patternFormationBar" and "g\_lockedPatternX" to -1 and 0, as the Gartley pattern requires X, A, B, C, and D points. Then, we proceed to extract the last five pivots from the "pivots" array, assigning "X" (earliest), "A", "B", "C", and "D" (latest) to form the pattern structure.

Next, we check for a bearish Gartley pattern (X high, A low, B high, C low, D high) by calculating the XA leg difference ("X.price - A.price"), ensuring it’s positive, computing the ideal B point as "A.price + 0.618 \* diff", verifying B is within "Tolerance \* diff" using [MathAbs](https://www.mql5.com/en/docs/math/mathabs), checking the BC leg (0.382 to 0.886 of AB), and confirming the AD retracement (0.786 of XA with D below X), setting "patternFound" to true and "patternType" to "Bearish" if valid. Last, we check for a bullish Gartley pattern (X low, A high, B low, C high, D low), calculating XA as "A.price - X.price", ensuring it’s positive, verifying B at 0.618 retracement, BC within 0.382 to 0.886 of AB, and AD at 0.786 of XA with D above X, setting "patternFound" to true and "patternType" to "Bullish" if valid. If the pattern is found, we can proceed to visualize it on the chart.

```
//--- If a valid Gartley pattern is detected
if(patternFound) {
   //--- Print a message indicating the pattern type and detection time
   Print(patternType, " Gartley pattern detected at ", TimeToString(D.time, TIME_DATE|TIME_MINUTES|TIME_SECONDS));

   //--- Create a unique prefix for all graphical objects related to this pattern
   string signalPrefix = "GA_" + IntegerToString(X.time);

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

If a valid pattern is detected ("patternFound" is true), we log the detection with [Print](https://www.mql5.com/en/docs/common/print), outputting the "patternType" ("Bullish" or "Bearish") and the D pivot’s time formatted with [TimeToString](https://www.mql5.com/en/docs/convert/timetostring), including date, minutes, and seconds. Then, we proceed to create a unique identifier "signalPrefix" by concatenating "GA\_" with "X.time" converted to a string using the [IntegerToString](https://www.mql5.com/en/docs/convert/integertostring) function. Next, we set "triangleColor" to blue for bullish patterns or red for bearish patterns. Last, we call "DrawTriangle" twice to visualize the pattern: first to draw the XAB triangle connecting pivots X, A, and B, and then to draw the BCD triangle connecting pivots B, C, and D, using "signalPrefix" with suffixes "\_Triangle1" and "\_Triangle2", respective pivot times and prices, "triangleColor", a width of 2, and enabling fill and background display with true flags. We get the following outcome.

![TRIANGLES SET](https://c.mql5.com/2/162/Screenshot_2025-08-08_193316.png)

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
      (patternType=="Bullish") ? "Bullish Gartley" : "Bearish Gartley");
   ObjectSetInteger(0, signalPrefix+"_Text_Center", OBJPROP_COLOR, clrBlack);
   ObjectSetInteger(0, signalPrefix+"_Text_Center", OBJPROP_FONTSIZE, 11);
   ObjectSetString(0, signalPrefix+"_Text_Center", OBJPROP_FONT, "Arial Bold");
   ObjectSetInteger(0, signalPrefix+"_Text_Center", OBJPROP_ALIGN, ALIGN_CENTER);
}
```

We further enhance the visualization of detected patterns by adding detailed chart objects to depict the pattern structure. First, we draw six solid trend lines using "DrawTrendLine" with the unique "signalPrefix" to connect key pivot points: XA, AB, BC, CD, XB, and BD, using pivot times and prices (e.g., "X.time", "X.price"), setting [OBJPROP\_COLOR](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer) to black, "OBJPROP\_WIDTH" to 2, and "OBJPROP\_STYLE" to "STYLE\_SOLID" with [ObjectSetInteger](https://www.mql5.com/en/docs/objects/objectsetinteger) to outline the pattern’s legs. Then, we proceed to retrieve the symbol’s point size with " [SymbolInfoDouble(\_Symbol, SYMBOL\_POINT)](https://www.mql5.com/en/docs/marketinformation/symbolinfodouble)" and calculate a 15-point offset for label positioning, determining Y-coordinates ("textY\_X", "textY\_A", "textY\_B", "textY\_C", "textY\_D") by adding or subtracting the offset based on whether each pivot is a swing high ("isHigh" true) or low to place labels above highs or below lows.

Next, we use "DrawTextEx" to create text labels for pivots X, A, B, C, and D with "signalPrefix" and suffixes like "\_Text\_X", displaying the respective letter, positioned at the pivot time and adjusted Y-coordinate, using "clrBlack", font size 11, and the pivot’s "isHigh" status for anchoring. Last, we calculate the central label’s position at "centralTime" as the midpoint of "X.time" and "B.time" and "centralPrice" at "D.price", creating a text object with [ObjectCreate](https://www.mql5.com/en/docs/objects/objectcreate) named "signalPrefix + '\_Text\_Center'", setting [OBJPROP\_TEXT](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_string) to "Bullish Gartley" or "Bearish Gartley" based on "patternType", and configuring "OBJPROP\_COLOR" to "clrBlack", "OBJPROP\_FONTSIZE" to 11, [OBJPROP\_FONT](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_string) to "Arial Bold", and "OBJPROP\_ALIGN" to "ALIGN\_CENTER" with the [ObjectSetString](https://www.mql5.com/en/docs/objects/objectsetstring) and "ObjectSetInteger" functions. This logic ensures a comprehensive visual representation of the pattern’s structure and type on the chart. When we run the program, here is a visualization of what we get.

![PATTERN WITH LABELS AND EDGES](https://c.mql5.com/2/162/Screenshot_2025-08-08_193447.png)

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

To define and visualize trade levels for the detected pattern, we set "lineStart" to the D pivot’s time ("D.time") and "lineEnd" to two periods ahead using " [PeriodSeconds(\_Period)](https://www.mql5.com/en/docs/common/periodseconds) \\* 2", and declare variables "entryPriceLevel", "TP1Level", "TP2Level", "TP3Level", and "tradeDiff" for trade calculations. Then, for a bullish pattern ("patternType == 'Bullish'"), we set "entryPriceLevel" to the current ask price with [SymbolInfoDouble](https://www.mql5.com/en/docs/marketinformation/symbolinfodouble), "TP3Level" to the C pivot’s price, calculate "tradeDiff" as "TP3Level - entryPriceLevel", and compute "TP1Level" and "TP2Level" as one-third and two-thirds of "tradeDiff" added to "entryPriceLevel"; for a bearish pattern, we use the bid price, set "TP3Level" to C’s price, calculate "tradeDiff" as "entryPriceLevel - TP3Level", and compute "TP1Level" and "TP2Level" by subtracting one-third and two-thirds of trade difference.

Next, we draw four dotted horizontal lines using "DrawDottedLine": an entry line at "entryPriceLevel" in magenta, and take-profit lines at "TP1Level" (forest green), "TP2Level" (green), and "TP3Level" (dark green), spanning from "lineStart" to "lineEnd". Last, we set "labelTime" to "lineEnd" plus half a period, create label texts with prices formatted via [DoubleToString](https://www.mql5.com/en/docs/convert/doubletostring) (e.g., "BUY (price)" or "SELL (price)" for entry, "TP1 (price)", etc.), and use "DrawTextEx" to draw these labels at "labelTime" with corresponding colors, font size 11, and anchored above the price levels, ensuring clear visualization of the Gartley pattern’s trade entry and take-profit levels. Upon compilation, we have the following outcome.

Bearish pattern:

![BEARISH PATTERN](https://c.mql5.com/2/162/Screenshot_2025-08-08_194225.png)

Bullish pattern:

![BULLISH PATTERN](https://c.mql5.com/2/162/Screenshot_2025-08-08_193731.png)

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
         double entryPriceTrade = 0, stopLoss = 0, takeProfit = 0;
         point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
         bool tradeResult = false;
         //--- For a Bullish pattern, execute a BUY trade
         if(patternType=="Bullish") {  //--- BUY signal
            entryPriceTrade = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
            double diffTrade = TP2Level - entryPriceTrade;
            stopLoss = entryPriceTrade - diffTrade * 3;
            takeProfit = TP2Level;
            tradeResult = obj_Trade.Buy(LotSize, _Symbol, entryPriceTrade, stopLoss, takeProfit, "Gartley Signal");
            if(tradeResult)
               Print("Buy order opened successfully.");
            else
               Print("Buy order failed: ", obj_Trade.ResultRetcodeDescription());
         }
         //--- For a Bearish pattern, execute a SELL trade
         else if(patternType=="Bearish") {  //--- SELL signal
            entryPriceTrade = SymbolInfoDouble(_Symbol, SYMBOL_BID);
            double diffTrade = entryPriceTrade - TP2Level;
            stopLoss = entryPriceTrade + diffTrade * 3;
            takeProfit = TP2Level;
            tradeResult = obj_Trade.Sell(LotSize, _Symbol, entryPriceTrade, stopLoss, takeProfit, "Gartley Signal");
            if(tradeResult)
               Print("Sell order opened successfully.");
            else
               Print("Sell order failed: ", obj_Trade.ResultRetcodeDescription());
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
}
else {
   //--- If no valid Gartley pattern is detected, reset the pattern lock variables
   g_patternFormationBar = -1;
   g_lockedPatternX = 0;
}
```

Here, we finalize the tick implementation by managing trade execution and pattern confirmation for the detected pattern. First, we retrieve the current bar index with " [Bars(\_Symbol, \_Period)](https://www.mql5.com/en/docs/series/bars) \- 1" and store it in "currentBarIndex". Then, if no pattern is locked ("g\_patternFormationBar == -1"), we set "g\_patternFormationBar" to "currentBarIndex", lock the X pivot time in "g\_lockedPatternX" with "X.time", log the detection indicating a wait for confirmation, and exit. Next, if still on the formation bar ("currentBarIndex == g\_patternFormationBar"), we log repainting and exit to prevent premature trading.

Last, if a new bar has formed ("currentBarIndex > g\_patternFormationBar") and the X pivot matches "g\_lockedPatternX", we confirm the pattern, log it, update "g\_patternFormationBar", and check if trading is permitted and no open positions exist via the [PositionSelect](https://www.mql5.com/en/docs/trading/positionselect) function; for a bullish pattern, we set "entryPriceTrade" to the ask price, calculate "diffTrade" as "TP2Level - entryPriceTrade", set "stopLoss" three times this distance below, set "takeProfit" to "TP2Level", and execute a buy with "obj\_Trade.Buy" using "LotSize" and "Gartley Signal", logging success or failure; for a bearish pattern, we use the bid price, set "stopLoss" three times above, and execute a sell with "obj\_Trade.Sell"; if trading is disallowed or a position exists, we log no trade; if the pattern changes, we update the lock and wait; if no pattern is found, we reset the global variables. Upon compilation, we have the following outcome.

Bearish signal:

![BEARISH SIGNAL](https://c.mql5.com/2/162/Screenshot_2025-08-08_194747.png)

Bullish signal:

![BULLISH SIGNAL](https://c.mql5.com/2/162/Screenshot_2025-08-08_194418.png)

From the image, we can see that we plot the harmonic pattern and are still able to trade it accordingly once it is confirmed, hence achieving our objective of identifying, plotting, and trading the pattern. The thing that remains is backtesting the program, and that is handled in the next section.

### Backtesting

After thorough backtesting, we have the following results.

Backtest graph:

![GRAPH](https://c.mql5.com/2/162/Screenshot_2025-08-08_230328.png)

Backtest report:

![REPORT](https://c.mql5.com/2/162/Screenshot_2025-08-08_230356.png)

### Conclusion

In conclusion, we’ve developed a [Gartley Pattern](https://www.mql5.com/go?link=https://www.investopedia.com/terms/g/gartley.asp "https://www.investopedia.com/terms/g/gartley.asp") system in MQL5 leveraging price action to detect bullish and bearish Gartley harmonic patterns with precise [Fibonacci ratios](https://www.metatrader5.com/en/terminal/help/objects/fibo/fibo_retracement "https://www.metatrader5.com/en/terminal/help/objects/fibo/fibo_retracement"), automating trades with calculated entry, stop loss, and multi-level take-profit points, and visualizing patterns with chart objects like triangles and trendlines.

Disclaimer: This article is for educational purposes only. Trading carries significant financial risks, and market volatility may result in losses. Thorough backtesting and careful risk management are crucial before deploying this program in live markets.

By leveraging the concepts and implementation presented, you can adapt this Gartley pattern system to your trading style, enhancing your algorithmic strategies. Happy trading!

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/19111.zip "Download all attachments in the single ZIP archive")

[Gartley\_Pattern\_EA.mq5](https://www.mql5.com/en/articles/download/19111/gartley_pattern_ea.mq5 "Download Gartley_Pattern_EA.mq5")(49.83 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/494255)**
(1)


![Miguel Angel Vico Alba](https://c.mql5.com/avatar/2025/10/68e99f33-714e.jpg)

**[Miguel Angel Vico Alba](https://www.mql5.com/en/users/mike_explosion)**
\|
30 Aug 2025 at 09:57

This article is a very good educational example for learning how to structure an EA and visualize harmonic patterns in MQL5.

Just one clarification for readers: although the introduction says "you'll have a powerful strategy ready for customization", this should not be taken as a trading-ready system.

The main reasons are:

- Incomplete Gartley definition: the code checks only that point D is around 0.786 of XA. In harmonic trading, the key idea is the Potential Reversal Zone (PRZ), which requires confluence of multiple ratios (for example 0.786 XA together with 1.27–1.618 of BC). Without that, many false patterns appear.
- Stop Loss / Take Profit rules: here they are defined as simple fractions of the move towards C. In harmonic trading the stop is usually placed beyond X, and targets are based on [Fibonacci retracements](https://www.mql5.com/en/docs/constants/objectconstants/enum_object "MQL5 documentation: Object Types") of AD. If this is not followed, the risk/reward becomes arbitrary.

So, as an exercise in coding and visualization, it's excellent. But for actual trading, a beginner should understand that more rules and validations are needed before relying on such a program.

![Neural Networks in Trading: A Multi-Agent Self-Adaptive Model (Final Part)](https://c.mql5.com/2/104/Multi-agent_adaptive_model_MASA___LOGO__1.png)[Neural Networks in Trading: A Multi-Agent Self-Adaptive Model (Final Part)](https://www.mql5.com/en/articles/16570)

In the previous article, we introduced the multi-agent self-adaptive framework MASA, which combines reinforcement learning approaches and self-adaptive strategies, providing a harmonious balance between profitability and risk in turbulent market conditions. We have built the functionality of individual agents within this framework. In this article, we will continue the work we started, bringing it to its logical conclusion.

![Building a Trading System (Part 3): Determining Minimum Risk Levels for Realistic Profit Targets](https://c.mql5.com/2/165/19141-building-a-trading-system-part-logo__1.png)[Building a Trading System (Part 3): Determining Minimum Risk Levels for Realistic Profit Targets](https://www.mql5.com/en/articles/19141)

Every trader's ultimate goal is profitability, which is why many set specific profit targets to achieve within a defined trading period. In this article, we will use Monte Carlo simulations to determine the optimal risk percentage per trade needed to meet trading objectives. The results will help traders assess whether their profit targets are realistic or overly ambitious. Finally, we will discuss which parameters can be adjusted to establish a practical risk percentage per trade that aligns with trading goals.

![Developing a Replay System (Part 78): New Chart Trade (V)](https://c.mql5.com/2/105/Desenvolvendo_um_sistema_de_Replay_Parte_77___LOGO.png)[Developing a Replay System (Part 78): New Chart Trade (V)](https://www.mql5.com/en/articles/12492)

In this article, we will look at how to implement part of the receiver code. Here we will implement an Expert Advisor to test and learn how the protocol interaction works. The content presented here is intended solely for educational purposes. Under no circumstances should the application be viewed for any purpose other than to learn and master the concepts presented.

![Multi-module trading robot in Python and MQL5 (Part I): Creating basic architecture and first modules](https://c.mql5.com/2/106/Multimodule_trading_robot_in_Python1_LOGO.png)[Multi-module trading robot in Python and MQL5 (Part I): Creating basic architecture and first modules](https://www.mql5.com/en/articles/16667)

We are going to develop a modular trading system that combines Python for data analysis with MQL5 for trade execution. Four independent modules monitor different market aspects in parallel: volumes, arbitrage, economics and risks, and use RandomForest with 400 trees for analysis. Particular emphasis is placed on risk management, since even the most advanced trading algorithms are useless without proper risk management.

[![](https://www.mql5.com/ff/si/s2n3m9ymjh52n07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F523%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dchoose.signals%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=fyznzyduwsltgnhlftytumasbfgbwlqw&s=91bc0eca8f132d3df7d14cdb1baebac753aef179403d60dc83856af55a4d6769&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=ulfkxhfbbjiiuyzpvvyljzvtggeqzyhq&ssn=1769091506311538721&ssn_dr=0&ssn_sr=0&fv_date=1769091506&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F19111&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Automating%20Trading%20Strategies%20in%20MQL5%20(Part%2029)%3A%20Creating%20a%20price%20action%20Gartley%20Harmonic%20Pattern%20system%20-%20MQL5%20Articles&scr_res=1920x1080&ac=17690915061232973&fz_uniq=5049034017312973745&sv=2552)

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