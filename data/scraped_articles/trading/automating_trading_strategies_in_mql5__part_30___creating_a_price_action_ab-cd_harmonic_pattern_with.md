---
title: Automating Trading Strategies in MQL5 (Part 30): Creating a Price Action AB-CD Harmonic Pattern with Visual Feedback
url: https://www.mql5.com/en/articles/19442
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 10
scraped_at: 2026-01-22T17:18:17.021920
---

[![](https://www.mql5.com/ff/si/6pp0j40fqxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=luckhiizjxvmvgigcufevttapwwrwbld&s=08cd1d929f27358481aded3c1c5f4e75a9bd5f52c477127afef2a5c532aec5c5&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=uyurxvsaddukvxneegroztozwpiikpsk&ssn=1769091495427833538&ssn_dr=0&ssn_sr=0&fv_date=1769091495&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F19442&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Automating%20Trading%20Strategies%20in%20MQL5%20(Part%2030)%3A%20Creating%20a%20Price%20Action%20AB-CD%20Harmonic%20Pattern%20with%20Visual%20Feedback%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909149539878714&fz_uniq=5049031951433704358&sv=2552)

MetaTrader 5 / Trading


### Introduction

In our [previous article (Part 29)](https://www.mql5.com/en/articles/19111), we developed a Gartley Pattern system in [MetaQuotes Language 5](https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5 "https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5") (MQL5) that detected bullish and bearish [Gartley harmonic patterns](https://www.mql5.com/go?link=https://www.investopedia.com/terms/g/gartley.asp "https://www.investopedia.com/terms/g/gartley.asp") using precise Fibonacci ratios, automating trades with calculated entry, stop loss, and take-profit levels, and visualizing patterns with chart objects such as triangles and trendlines. In Part 30, we introduce an AB=CD Pattern system. While the Gartley system depends on detecting specific multi-leg harmonic structures defined by multiple Fibonacci levels, the AB=CD system specifically identifies patterns formed when two equivalent price segments (AB and CD) are found through pivot points and distinct retracement and extension ratios—resulting in simpler yet dynamic pattern identification. The AB=CD system executes trades using dynamic entries and multi-level take-profit targets, enhancing visualization with triangles, trendlines, and labels for clear pattern presentation. We will cover the following topics:

1. [Understanding the AB=CD Harmonic Pattern Framework](https://www.mql5.com/en/articles/19442#para1)
2. [Implementation in MQL5](https://www.mql5.com/en/articles/19442#para2)
3. [Backtesting](https://www.mql5.com/en/articles/19442#para3)
4. [Conclusion](https://www.mql5.com/en/articles/19442#para4)

By the end, you’ll have a robust MQL5 strategy for AB=CD harmonic pattern trading, ready for customization—let’s dive in!

### Understanding the AB=CD Harmonic Pattern Framework

The [AB=CD pattern](https://www.mql5.com/go?link=https://harmonictrader.com/harmonic-patterns/abcd-pattern/ "https://harmonictrader.com/harmonic-patterns/abcd-pattern/") is a harmonic trading formation that identifies potential reversal zones through four key swing points—A, B, C, and D—existing in bullish and bearish forms, leveraging [Fibonacci ratios](https://www.metatrader5.com/en/terminal/help/objects/fibo/fibo_retracement "https://www.metatrader5.com/en/terminal/help/objects/fibo/fibo_retracement") to pinpoint high-probability trade setups. In a bullish AB=CD, the structure forms a high-low-high-low sequence where A is a swing high, B a swing low, C a swing high, and D a swing low (below B), with the AB and CD legs being equal in length or related by Fibonacci retracement and extension ratios; a bearish AB=CD follows a low-high-low-high sequence, with D above B. Here is a visualization of the patterns:

Bearish Harmonic AB=CD pattern:

![BEARISH HARMONIC AB=CD PATTERN](https://c.mql5.com/2/167/Screenshot_2025-09-06_120156.png)

Bullish Harmonic AB=CD pattern:

![BULLISH HARMONIC AB=CD PATTERN](https://c.mql5.com/2/167/Screenshot_2025-09-06_120139.png)

Our approach involves detecting these swing pivots within a specified bar range, validating the pattern by ensuring the BC leg retraces 0.382 to 0.886 of AB and the CD leg extends 1.13 to 2.618 of BC, visualizing the pattern with chart objects like triangles and trendlines for clarity, and executing trades at the D point with calculated stop loss and multiple take-profit levels based on Fibonacci retracements to capitalize on anticipated reversals. Let’s proceed to the implementation!

### Implementation in MQL5

To create the program in MQL5, open the [MetaEditor](https://www.metatrader5.com/en/automated-trading/metaeditor "https://www.metatrader5.com/en/automated-trading/metaeditor"), go to the Navigator, locate the Indicators folder, click on the "New" tab, and follow the prompts to create the file. Once it is made, in the coding environment, we will need to declare some [global variables](https://www.mql5.com/en/docs/basis/variables/global) that we will use throughout the program.

```
//+------------------------------------------------------------------+
//|                                              ABCD Pattern EA.mq5 |
//|                        Copyright 2025, Forex Algo-Trader, Allan. |
//|                                 "https://t.me/Forex_Algo_Trader" |
//+------------------------------------------------------------------+
#property copyright "Forex Algo-Trader, Allan"
#property link "https://t.me/Forex_Algo_Trader"
#property version "1.00"
#property description "This EA trades based on AB=CD Strategy"
#property strict

//--- Include the trading library for order functions
#include <Trade\Trade.mqh> //--- Include Trade library
CTrade obj_Trade; //--- Instantiate a obj_Trade object

//--- Input parameters for user configuration
input int PivotLeft = 5;          // Number of bars to the left for pivot check
input int PivotRight = 5;         // Number of bars to the right for pivot check
input double Tolerance = 0.10;    // Allowed deviation (10% of AB move)
input double LotSize = 0.01;      // Lot size for new orders
input bool AllowTrading = true;   // Enable or disable trading
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
datetime g_lockedPatternA = 0;  //--- The key A pivot time for the locked pattern
```

To lay the foundation for the [AB=CD pattern](https://www.mql5.com/go?link=https://harmonictrader.com/harmonic-patterns/abcd-pattern/ "https://harmonictrader.com/harmonic-patterns/abcd-pattern/"), first, we include the "<Trade\\Trade.mqh>" library and instantiate "obj\_Trade" as a [CTrade](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade) object to manage trade operations, such as executing buy and sell orders. Then, we proceed to define [input parameters](https://www.mql5.com/en/docs/basis/variables/inputvariables) for user customization: "PivotLeft" and "PivotRight" at 5 bars each to set the lookback range for pivot detection, "Tolerance" at 0.10 to allow a 10% deviation in [Fibonacci ratios](https://www.metatrader5.com/en/terminal/help/objects/fibo/fibo_retracement "https://www.metatrader5.com/en/terminal/help/objects/fibo/fibo_retracement"), "LotSize" at 0.01 for trade volume, and "AllowTrading" as true to enable automated trading.

Next, we define the "Pivot" [structure](https://www.mql5.com/en/docs/basis/types/classes) with "time" (datetime), "price" ( [double](https://www.mql5.com/en/docs/basis/types/double)), and "isHigh" (bool) to store swing points, declare "pivots" as a dynamic array to hold these points, and initialize globals "g\_patternFormationBar" to -1 to track the bar where a pattern forms and "g\_lockedPatternA" to 0 to lock the A pivot time for pattern confirmation, noting the use of A instead of X to align with the AB=CD pattern’s focus on the AB and CD legs. This setup establishes the core framework for detecting and trading AB=CD patterns. For visualization, we can have functions to draw lines, labels, and triangles.

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
//| Helper: Draw a dotted trend line |
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

We proceed to implement visualization functions to create clear chart representations of the AB=CD harmonic pattern and its trade levels. First, we develop the "DrawTriangle" function, which uses [ObjectCreate](https://www.mql5.com/en/docs/objects/objectcreate) to draw a filled triangle ( [OBJ\_TRIANGLE](https://www.mql5.com/en/docs/constants/objectconstants/enum_object/obj_triangle)) defined by three points with times ("t1", "t2", "t3") and prices ("p1", "p2", "p3"), setting [OBJPROP\_COLOR](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer) to the specified color, "OBJPROP\_STYLE" to "STYLE\_SOLID", "OBJPROP\_WIDTH" to the given width, "OBJPROP\_FILL" to enable or disable filling, and "OBJPROP\_BACK" to set background or foreground placement using the [ObjectSetInteger](https://www.mql5.com/en/docs/objects/objectsetinteger) function.

Then, we proceed to create the "DrawTrendLine" function, which draws a trend line ( [OBJ\_TREND](https://www.mql5.com/en/docs/constants/objectconstants/enum_object/obj_trend)) between two points and last, we develop the "DrawTextEx" function, which creates a text label ( [OBJ\_TEXT](https://www.mql5.com/en/docs/constants/objectconstants/enum_object/obj_text)) at coordinates ("t", "p") with "ObjectCreate", setting "OBJPROP\_TEXT" to the specified text, "OBJPROP\_COLOR", "OBJPROP\_FONTSIZE", and "OBJPROP\_FONT" to "Arial Bold" using [ObjectSetString](https://www.mql5.com/en/docs/objects/objectsetstring) and "ObjectSetInteger", anchoring above for swing highs or below for lows based on "isHigh" with [OBJPROP\_ANCHOR](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer), and centering with "OBJPROP\_ALIGN". We can now proceed to the [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) event handler and attempt to identify pivot points that we can use later for pattern recognition. Here is the logic we use to achieve that.

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

Here, we implement the initial logic of the [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) function. First, we declare a static "lastBarTime" initialized to 0 to track the last processed bar and compare it with "currentBarTime" obtained from [iTime](https://www.mql5.com/en/docs/series/itime) at shift 1 for the current symbol and period, exiting if unchanged to avoid redundant processing, and updating "lastBarTime" when a new bar is detected. Then, we proceed to clear the "pivots" array with [ArrayResize](https://www.mql5.com/en/docs/array/arrayresize)" to ensure a fresh analysis. Next, we retrieve the total number of bars with [Bars](https://www.mql5.com/en/docs/series/bars), set the pivot detection range with "start" as "PivotLeft" and "end" as total bars minus "PivotRight", and iterate through bars from "end - 1" to start.

For each bar, we assume it’s a swing high ("isPivotHigh" true) and low ("isPivotLow" true), obtain its high and low prices using "iHigh" and "iLow", and validate the pivot by checking surrounding bars within "PivotLeft" and "PivotRight" with [iHigh](https://www.mql5.com/en/docs/series/ihigh) and [iLow](https://www.mql5.com/en/docs/series/ilow), invalidating the pivot if any neighboring bar has a higher high or lower low. Last, if the bar qualifies as a pivot, we create a "Pivot" [structure](https://www.mql5.com/en/docs/basis/types/classes), set its "time" with "iTime", "price" to the high or low based on "isPivotHigh", and "isHigh" flag, then append it to the "pivots" array using [ArrayResize](https://www.mql5.com/en/docs/array/arrayresize) and store it. We get the following array of data when we print the pivot structure.

![PIVOTS DATA](https://c.mql5.com/2/167/Screenshot_2025-09-06_105517.png)

With the data, we can extract the pivot points, and if we have enough pivots, we can analyze and detect the patterns. Here is the logic we implement to achieve that.

```
//--- Determine the total number of pivots found
int pivotCount = ArraySize(pivots);
//--- If fewer than four pivots are found, the pattern cannot be formed
if(pivotCount < 4) {
   //--- Reset pattern lock variables
   g_patternFormationBar = -1;
   g_lockedPatternA = 0;
   //--- Exit the OnTick function
   return;
}

//--- Extract the last four pivots as A, B, C, and D
Pivot A = pivots[pivotCount - 4];
Pivot B = pivots[pivotCount - 3];
Pivot C = pivots[pivotCount - 2];
Pivot D = pivots[pivotCount - 1];

//--- Initialize a flag to indicate if a valid AB=CD pattern is found
bool patternFound = false;
//--- Initialize pattern type
string patternType = "";
double used_retr = 0.0;
double used_ext = 0.0;
//--- Check for the high-low-high-low (Bullish reversal) structure
if(A.isHigh && (!B.isHigh) && C.isHigh && (!D.isHigh)) {
   //--- Calculate the difference between pivot A and B
   double diff = A.price - B.price;
   //--- Ensure the difference is positive
   if(diff > 0) {
      //--- Calculate the BC leg length
      double BC = C.price - B.price;
      double retrace = BC / diff;
      //--- Calculate the CD leg length
      double CD = C.price - D.price;
      double extension = CD / BC;
      //--- Define fib ratios
      double fib_retr[] = {0.382, 0.5, 0.618, 0.786, 0.886};
      double fib_ext[] = {2.618, 2.0, 1.618, 1.272, 1.13};
      bool valid = false;
      for(int k = 0; k < ArraySize(fib_retr); k++) {
         if(MathAbs(retrace - fib_retr[k]) <= Tolerance && MathAbs(extension - fib_ext[k]) <= Tolerance) {
            valid = true;
            used_retr = fib_retr[k];
            used_ext = fib_ext[k];
            break;
         }
      }
      if(valid && (D.price < B.price)) {
         patternFound = true;
         patternType = "Bullish";
      }
   }
}
//--- Check for the low-high-low-high (Bearish reversal) structure
if((!A.isHigh) && B.isHigh && (!C.isHigh) && D.isHigh) {
   //--- Calculate the difference between pivot B and A
   double diff = B.price - A.price;
   //--- Ensure the difference is positive
   if(diff > 0) {
      //--- Calculate the BC leg length
      double BC = B.price - C.price;
      double retrace = BC / diff;
      //--- Calculate the CD leg length
      double CD = D.price - C.price;
      double extension = CD / BC;
      //--- Define fib ratios
      double fib_retr[] = {0.382, 0.5, 0.618, 0.786, 0.886};
      double fib_ext[] = {2.618, 2.0, 1.618, 1.272, 1.13};
      bool valid = false;
      for(int k = 0; k < ArraySize(fib_retr); k++) {
         if(MathAbs(retrace - fib_retr[k]) <= Tolerance && MathAbs(extension - fib_ext[k]) <= Tolerance) {
            valid = true;
            used_retr = fib_retr[k];
            used_ext = fib_ext[k];
            break;
         }
      }
      if(valid && (D.price > B.price)) {
         patternFound = true;
         patternType = "Bearish";
      }
   }
}
```

First, we determine the total number of pivots with " [ArraySize(pivots)](https://www.mql5.com/en/docs/array/arraysize)" stored in "pivotCount" and exit if fewer than 4 pivots are found, resetting "g\_patternFormationBar" and "g\_lockedPatternA" to -1 and 0, as the AB=CD pattern requires A, B, C, and D points. Then, we proceed to extract the last four pivots from the "pivots" array, assigning "A" (earliest), "B", "C", and "D" (latest) to form the pattern structure.

Next, we check for a bullish AB=CD pattern (A high, B low, C high, D low) by calculating the AB leg difference ("A.price - B.price"), ensuring it’s positive, computing the BC leg length ("C.price - B.price") and its retracement ratio relative to AB, calculating the CD leg length ("C.price - D.price") and its extension ratio relative to BC, defining Fibonacci retracement ratios (0.382, 0.5, 0.618, 0.786, 0.886) and extension ratios (2.618, 2.0, 1.618, 1.272, 1.13), and validating if both ratios fall within "Tolerance" while ensuring "D.price < B.price", setting "patternFound" to true, "patternType" to "Bullish", and storing the matched "used\_retr" and "used\_ext". Last, we check for a bearish AB=CD pattern (A low, B high, C low, D high) using similar calculations for AB ("B.price - A.price"), BC ("B.price - C.price"), and CD ("D.price - C.price"), validating against the same Fibonacci ratios and ensuring "D.price > B.price", setting "patternFound" to true and "patternType" to "Bearish" if valid. If the pattern is found, we can proceed to visualize it on the chart.

```
//--- If a valid AB=CD pattern is detected
if(patternFound) {
   //--- Print a message indicating the pattern type and detection time
   Print(patternType, " AB=CD pattern detected at ", TimeToString(D.time, TIME_DATE|TIME_MINUTES|TIME_SECONDS));

   //--- Create a unique prefix for all graphical objects related to this pattern
   string signalPrefix = "AB_" + IntegerToString(A.time);

   //--- Choose triangle color based on the pattern type
   color triangleColor = (patternType=="Bullish") ? clrBlue : clrRed;

   //--- Draw the first triangle connecting pivots A, B, and C
   DrawTriangle(signalPrefix+"_Triangle1", A.time, A.price, B.time, B.price, C.time, C.price,
                triangleColor, 2, true, true);
   //--- Draw the second triangle connecting pivots B, C, and D
   DrawTriangle(signalPrefix+"_Triangle2", B.time, B.price, C.time, C.price, D.time, D.price,
                triangleColor, 2, true, true);
}
```

To visualize the pattern, first, if a valid pattern is detected ("patternFound" is true), we log the detection with [Print](https://www.mql5.com/en/docs/common/print), outputting the "patternType" ("Bullish" or "Bearish") and the D pivot’s time formatted with [TimeToString](https://www.mql5.com/en/docs/convert/timetostring), including date, minutes, and seconds. Then, we proceed to create a unique identifier "signalPrefix" by concatenating "AB\_" with "A.time" converted to a string using [IntegerToString](https://www.mql5.com/en/docs/convert/integertostring) to ensure distinct naming for chart objects.

Next, we set "triangleColor" to blue for bullish patterns or red for bearish patterns to differentiate them visually. Last, we call "DrawTriangle" twice to visualize the pattern: first to draw the ABC triangle connecting pivots A, B, and C, and then to draw the BCD triangle connecting pivots B, C, and D, using "signalPrefix" with suffixes "\_Triangle1" and "\_Triangle2", respective pivot times and prices, "triangleColor", a width of 2, and enabling fill and background display with true flags. We get the following outcome.

![TRIANGLE SET](https://c.mql5.com/2/167/Screenshot_2025-09-06_110734.png)

From the image, we can see that we can map and visualize the detected pattern correctly. We now need to continue mapping the trendlines to fully make it visible within boundaries and add a label to it for easier identification of the levels.

```
//--- Draw boundary trend lines connecting the pivots for clarity
DrawTrendLine(signalPrefix+"_TL_AB", A.time, A.price, B.time, B.price, clrBlack, 2, STYLE_SOLID);
DrawTrendLine(signalPrefix+"_TL_BC", B.time, B.price, C.time, C.price, clrBlack, 2, STYLE_SOLID);
DrawTrendLine(signalPrefix+"_TL_CD", C.time, C.price, D.time, D.price, clrBlack, 2, STYLE_SOLID);

//--- Retrieve the symbol's point size to calculate offsets for text positioning
double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
//--- Calculate an offset (15 points) for positioning text above or below pivots
double offset = 15 * point;

//--- Determine the Y coordinate for each pivot label based on its type
double textY_A = (A.isHigh ? A.price + offset : A.price - offset);
double textY_B = (B.isHigh ? B.price + offset : B.price - offset);
double textY_C = (C.isHigh ? C.price + offset : C.price - offset);
double textY_D = (D.isHigh ? D.price + offset : D.price - offset);

//--- Draw text labels for each pivot with appropriate anchoring
DrawTextEx(signalPrefix+"_Text_A", "A", A.time, textY_A, clrBlack, 11, A.isHigh);
DrawTextEx(signalPrefix+"_Text_B", "B", B.time, textY_B, clrBlack, 11, B.isHigh);
DrawTextEx(signalPrefix+"_Text_C", "C", C.time, textY_C, clrBlack, 11, C.isHigh);
DrawTextEx(signalPrefix+"_Text_D", "D", D.time, textY_D, clrBlack, 11, D.isHigh);

//--- Calculate the central label's time as the midpoint between pivots A and C
datetime centralTime = (A.time + C.time) / 2;
//--- Set the central label's price at pivot D's price
double centralPrice = D.price;
//--- Create the central text label indicating the pattern type
if(ObjectCreate(0, signalPrefix+"_Text_Center", OBJ_TEXT, 0, centralTime, centralPrice)) {
   ObjectSetString(0, signalPrefix+"_Text_Center", OBJPROP_TEXT,
      (patternType=="Bullish") ? "Bullish AB=CD" : "Bearish AB=CD");
   ObjectSetInteger(0, signalPrefix+"_Text_Center", OBJPROP_COLOR, clrBlack);
   ObjectSetInteger(0, signalPrefix+"_Text_Center", OBJPROP_FONTSIZE, 11);
   ObjectSetString(0, signalPrefix+"_Text_Center", OBJPROP_FONT, "Arial Bold");
   ObjectSetInteger(0, signalPrefix+"_Text_Center", OBJPROP_ALIGN, ALIGN_CENTER);
}
```

We further enhance the visualization of detected patterns by adding detailed chart objects to clearly depict the pattern structure. First, we draw three solid trend lines using "DrawTrendLine" with the unique "signalPrefix" to connect key pivot points: AB, BC, and CD, using pivot times and prices (e.g., "A.time", "A.price"), setting [OBJPROP\_COLOR](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer) to "clrBlack", "OBJPROP\_WIDTH" to 2, and "OBJPROP\_STYLE" to "STYLE\_SOLID" with [ObjectSetInteger](https://www.mql5.com/en/docs/objects/objectsetinteger) to outline the pattern’s legs. Then, we proceed to retrieve the symbol’s point size with [SymbolInfoDouble(\_Symbol, SYMBOL\_POINT)](https://www.mql5.com/en/docs/marketinformation/symbolinfodouble) and calculate a 15-point offset for label positioning, determining Y-coordinates ("textY\_A", "textY\_B", "textY\_C", "textY\_D") by adding or subtracting the offset based on whether each pivot is a swing high ("isHigh" true) or low to place labels above highs or below lows.

Next, we use "DrawTextEx" to create text labels for pivots A, B, C, and D with "signalPrefix" and suffixes like "\_Text\_A", displaying the respective letter, positioned at the pivot time and adjusted Y-coordinate, using "clrBlack", font size 11, and the pivot’s "isHigh" status for anchoring. Last, we calculate the central label’s position at "centralTime" as the midpoint of "A.time" and "C.time" and "centralPrice" at "D.price", creating a text object with [ObjectCreate](https://www.mql5.com/en/docs/objects/objectcreate) named "signalPrefix + '\_Text\_Center'", setting [OBJPROP\_TEXT](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_string) to "Bullish AB=CD" or "Bearish AB=CD" based on "patternType", and configuring "OBJPROP\_COLOR" to "clrBlack", "OBJPROP\_FONTSIZE" to 11, [OBJPROP\_FONT](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_string) to "Arial Bold", and "OBJPROP\_ALIGN" to "ALIGN\_CENTER" with [ObjectSetString](https://www.mql5.com/en/docs/objects/objectsetstring) and "ObjectSetInteger", ensuring a comprehensive visual representation of the pattern’s structure and type on the chart. When we run the program, here is a visualization of the output we receive.

![PATTERN WITH LABELS AND EDGES](https://c.mql5.com/2/167/Screenshot_2025-09-06_111551.png)

From the image, we can see that we have added the edges and the labels to the pattern, making it more revealing and illustrative. What we need to do next is determine the trade levels for the pattern.

```
//--- Define start and end times for drawing horizontal dotted lines for trade levels
datetime lineStart = D.time;
datetime lineEnd = D.time + PeriodSeconds(_Period)*2;

//--- Declare variables for entry price and take profit levels
double entryPriceLevel, TP1Level, TP2Level, TP3Level;
//--- Calculate pattern range (CD length)
double patternRange = (patternType=="Bullish") ? (C.price - D.price) : (D.price - C.price);
//--- Calculate trade levels based on whether the pattern is Bullish or Bearish
if(patternType=="Bullish") { //--- Bullish → BUY signal
   //--- Use the current ASK price as the entry
   entryPriceLevel = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   //--- Set TP3 at pivot C's price
   TP3Level = C.price;
   //--- Set TP1 at 0.382 fib retrace from D to C
   TP1Level = D.price + 0.382 * patternRange;
   //--- Set TP2 at 0.618 fib retrace from D to C
   TP2Level = D.price + 0.618 * patternRange;
} else { //--- Bearish → SELL signal
   //--- Use the current BID price as the entry
   entryPriceLevel = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   //--- Set TP3 at pivot C's price
   TP3Level = C.price;
   //--- Set TP1 at 0.382 fib retrace from D to C
   TP1Level = D.price - 0.382 * patternRange;
   //--- Set TP2 at 0.618 fib retrace from D to C
   TP2Level = D.price - 0.618 * patternRange;
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

To define and visualize trade levels, we set "lineStart" to the D pivot’s time ("D.time") and "lineEnd" to two periods ahead using " [PeriodSeconds(\_Period)](https://www.mql5.com/en/docs/common/periodseconds) \\* 2", and declare variables "entryPriceLevel", "TP1Level", "TP2Level", and "TP3Level" for trade calculations. Then, we calculate the "patternRange" as the CD leg length ("C.price - D.price" for bullish, "D.price - C.price" for bearish); for a bullish pattern, we set "entryPriceLevel" to the ask price with [SymbolInfoDouble](https://www.mql5.com/en/docs/marketinformation/symbolinfodouble), "TP3Level" to C’s price, "TP1Level" to "D.price + 0.382 \* patternRange", and "TP2Level" to "D.price + 0.618 \* patternRange"; for a bearish pattern, we use the bid price, set "TP3Level" to C’s price, "TP1Level" to "D.price - 0.382 \* patternRange", and "TP2Level" to "D.price - 0.618 \* patternRange".

Next, we draw four dotted horizontal lines using "DrawDottedLine": an entry line at "entryPriceLevel" in magenta, and take-profit lines at "TP1Level" (forest green), "TP2Level" (green), and "TP3Level" (dark green), spanning from "lineStart" to "lineEnd". Last, we set "labelTime" to "lineEnd" plus half a period, create label texts with prices formatted via [DoubleToString](https://www.mql5.com/en/docs/convert/doubletostring) (e.g., "BUY (price)" or "SELL (price)" for entry, "TP1 (price)", etc.), and use "DrawTextEx" to draw these labels at "labelTime" with corresponding colors, font size 11, and anchored above the price levels. Upon compilation, we have the following outcome.

Bearish pattern:

![BEARISH PATTERN](https://c.mql5.com/2/167/Screenshot_2025-09-06_112445.png)

Bullish pattern:

![BULLISH PATTERN](https://c.mql5.com/2/167/Screenshot_2025-09-06_112607.png)

From the images, we can see that we have correctly mapped the trade levels. What we need to do now is initiate the actual trade positions, and that is all.

```
//--- Retrieve the index of the current bar
int currentBarIndex = Bars(_Symbol, _Period) - 1;
//--- If no pattern has been previously locked, lock the current pattern formation
if(g_patternFormationBar == -1) {
   g_patternFormationBar = currentBarIndex;
   g_lockedPatternA = A.time;
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
   //--- Check if the locked pattern still corresponds to the same A pivot
   if(g_lockedPatternA == A.time) {
      Print("Confirmed pattern (locked on bar ", g_patternFormationBar, "). Opening trade on bar ", currentBarIndex, ".");
      //--- Update the pattern formation bar to the current bar
      g_patternFormationBar = currentBarIndex;
      //--- Only proceed with trading if allowed and if there is no existing position
      if(AllowTrading && !PositionSelect(_Symbol)) {
         double entryPriceTrade = 0, stopLoss = 0, takeProfit = 0;
         point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
         bool tradeResult = false;
         //--- Determine next extension for SL
         double next_ext = 0.0;
         if(MathAbs(used_ext - 1.13) < 0.05) next_ext = 1.272;
         else if(MathAbs(used_ext - 1.272) < 0.05) next_ext = 1.618;
         else if(MathAbs(used_ext - 1.618) < 0.05) next_ext = 2.0;
         else if(MathAbs(used_ext - 2.0) < 0.05) next_ext = 2.618;
         else if(MathAbs(used_ext - 2.618) < 0.05) next_ext = 3.618;
         else next_ext = used_ext * 1.618; // fallback
         //--- For a Bullish pattern, execute a BUY trade
         if(patternType=="Bullish") { //--- BUY signal
            entryPriceTrade = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
            double BC_leg = C.price - B.price;
            stopLoss = C.price - next_ext * BC_leg;
            if(stopLoss > D.price) stopLoss = D.price - 10 * point;
            takeProfit = TP2Level;
            tradeResult = obj_Trade.Buy(LotSize, _Symbol, entryPriceTrade, stopLoss, takeProfit, "AB=CD Signal");
            if(tradeResult)
               Print("Buy order opened successfully.");
            else
               Print("Buy order failed: ", obj_Trade.ResultRetcodeDescription());
         }
         //--- For a Bearish pattern, execute a SELL trade
         else if(patternType=="Bearish") { //--- SELL signal
            entryPriceTrade = SymbolInfoDouble(_Symbol, SYMBOL_BID);
            double BC_leg = B.price - C.price;
            stopLoss = C.price + next_ext * BC_leg;
            if(stopLoss < D.price) stopLoss = D.price + 10 * point;
            takeProfit = TP2Level;
            tradeResult = obj_Trade.Sell(LotSize, _Symbol, entryPriceTrade, stopLoss, takeProfit, "AB=CD Signal");
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
      //--- If the pattern has changed, update the lock with the new formation bar and A pivot
      g_patternFormationBar = currentBarIndex;
      g_lockedPatternA = A.time;
      Print("Pattern has changed; updating lock on bar ", currentBarIndex, ". Waiting for confirmation.");
      return;
   }
}
}
else {
   //--- If no valid AB=CD pattern is detected, reset the pattern lock variables
   g_patternFormationBar = -1;
   g_lockedPatternA = 0;
}
```

Here, we finalize the tick implementation for the pattern by managing trade execution and pattern confirmation for the detected AB=CD harmonic pattern. First, we retrieve the current bar index with " [Bars(\_Symbol, \_Period)](https://www.mql5.com/en/docs/series/bars) \- 1" and store it in "currentBarIndex". Then, if no pattern is locked ("g\_patternFormationBar == -1"), we set "g\_patternFormationBar" to "currentBarIndex", lock the A pivot time in "g\_lockedPatternA" with "A.time", log the detection indicating a wait for confirmation, and exit. Next, if still on the formation bar ("currentBarIndex == g\_patternFormationBar"), we exit to prevent premature trading.

Last, if a new bar has formed ("currentBarIndex > g\_patternFormationBar") and the A pivot matches "g\_lockedPatternA", we confirm the pattern, log it, update "g\_patternFormationBar", and check if trading is permitted with "AllowTrading" and no open positions exist via [PositionSelect](https://www.mql5.com/en/docs/trading/positionselect); we determine the next Fibonacci extension ("next\_ext") for stop loss based on "used\_ext" (e.g., 1.13 to 1.272, up to 3.618, or a fallback of "used\_ext \* 1.618"); for a bullish pattern, we set "entryPriceTrade" to the ask price, calculate "BC\_leg" as "C.price - B.price", set "stopLoss" to "C.price - next\_ext \* BC\_leg" (adjusted to "D.price - 10 \* point" if above D), set "takeProfit" to "TP2Level", and execute a buy with "obj\_Trade.Buy" using "LotSize" and "AB=CD Signal", logging success or failure; for a bearish pattern, we use the bid price, calculate "BC\_leg" as "B.price - C.price", set "stopLoss" to "C.price + next\_ext \* BC\_leg" (adjusted to "D.price + 10 \* point" if below D), and execute a sell with "obj\_Trade.Sell"; if trading is disallowed or a position exists, we log no trade; if the pattern changes, we update the lock and wait; if no pattern is found, we reset the global variables. Upon compilation, we have the following outcome.

Bearish signal:

![BEARISH SIGNAL](https://c.mql5.com/2/167/Screenshot_2025-09-06_113713.png)

Bullish signal:

![BULLISH SIGNAL](https://c.mql5.com/2/167/Screenshot_2025-09-06_113830.png)

From the image, we can see that we plot the harmonic pattern and are still able to trade it accordingly once it is confirmed, hence achieving our objective of identifying, plotting, and trading the pattern. The thing that remains is backtesting the program, and that is handled in the next section.

### Backtesting

After thorough backtesting, we have the following results.

Backtest graph:

![GRAPH](https://c.mql5.com/2/167/Screenshot_2025-09-06_125937.png)

Backtest report:

![REPORT](https://c.mql5.com/2/167/Screenshot_2025-09-06_130008.png)

### Conclusion

In conclusion, we’ve developed an AB=CD pattern system in [MQL5](https://www.mql5.com/), leveraging price action to detect bullish and bearish AB=CD harmonic patterns with precise Fibonacci retracement and extension ratios, automating trades with calculated entry, stop loss, and multi-level take-profit points, and visualizing patterns with chart objects like triangles and trendlines.

Disclaimer: This article is for educational purposes only. Trading carries significant financial risks, and market volatility may result in losses. Thorough backtesting and careful risk management are crucial before deploying this program in live markets.

By leveraging the concepts and implementation presented, you can adapt this AB=CD pattern system to your trading style, enhancing your algorithmic strategies. Happy trading!

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/19442.zip "Download all attachments in the single ZIP archive")

[ABdCD\_Pattern\_EA.mq5](https://www.mql5.com/en/articles/download/19442/ABdCD_Pattern_EA.mq5 "Download ABdCD_Pattern_EA.mq5")(46.42 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/495119)**

![Moving to MQL5 Algo Forge (Part 1): Creating the Main Repository](https://c.mql5.com/2/130/Moving_to_MQL5_Algo_Forge_Part_LOGO__3.png)[Moving to MQL5 Algo Forge (Part 1): Creating the Main Repository](https://www.mql5.com/en/articles/17646)

When working on projects in MetaEditor, developers often face the need to manage code versions. MetaQuotes recently announced migration to GIT and the launch of MQL5 Algo Forge with code versioning and collaboration capabilities. In this article, we will discuss how to use the new and previously existing tools more efficiently.

![From Novice to Expert: Animated News Headline Using MQL5 (X)—Multiple Symbol Chart View for News Trading](https://c.mql5.com/2/168/19299-from-novice-to-expert-animated-logo.png)[From Novice to Expert: Animated News Headline Using MQL5 (X)—Multiple Symbol Chart View for News Trading](https://www.mql5.com/en/articles/19299)

Today we will develop a multi-chart view system using chart objects. The goal is to enhance news trading by applying MQL5 algorithms that help reduce trader reaction time during periods of high volatility, such as major news releases. In this case, we provide traders with an integrated way to monitor multiple major symbols within a single all-in-one news trading tool. Our work is continuously advancing with the News Headline EA, which now features a growing set of functions that add real value both for traders using fully automated systems and for those who prefer manual trading assisted by algorithms. Explore more knowledge, insights, and practical ideas by clicking through and joining this discussion.

![Elevate Your Trading With Smart Money Concepts (SMC): OB, BOS, and FVG](https://c.mql5.com/2/168/16340-elevate-your-trading-with-smart-logo.png)[Elevate Your Trading With Smart Money Concepts (SMC): OB, BOS, and FVG](https://www.mql5.com/en/articles/16340)

Elevate your trading with Smart Money Concepts (SMC) by combining Order Blocks (OB), Break of Structure (BOS), and Fair Value Gaps (FVG) into one powerful EA. Choose automatic strategy execution or focus on any individual SMC concept for flexible and precise trading.

![Price Action Analysis Toolkit Development (Part 39): Automating BOS and ChoCH Detection in MQL5](https://c.mql5.com/2/168/19365-price-action-analysis-toolkit-logo.png)[Price Action Analysis Toolkit Development (Part 39): Automating BOS and ChoCH Detection in MQL5](https://www.mql5.com/en/articles/19365)

This article presents Fractal Reaction System, a compact MQL5 system that converts fractal pivots into actionable market-structure signals. Using closed-bar logic to avoid repainting, the EA detects Change-of-Character (ChoCH) warnings and confirms Breaks-of-Structure (BOS), draws persistent chart objects, and logs/alerts every confirmed event (desktop, mobile and sound). Read on for the algorithm design, implementation notes, testing results and the full EA code so you can compile, test and deploy the detector yourself.

[![](https://www.mql5.com/ff/si/w766tj9vyj3g607n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fmarket%2Fmt5%2Fexpert%3FHasRent%3Don%26utm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Drent.expert%26utm_content%3Drent.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=sorsafcerhkgwrjzwwrpvelbicxjwzon&s=ae91b1eae8acb61167455495742e6cc8eb55ccedb33fd953f8256b68cbe9c3b4&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=irjibwfyycdjtuyhbduhdvudojtfzbls&ssn=1769091495427833538&ssn_dr=0&ssn_sr=0&fv_date=1769091495&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F19442&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Automating%20Trading%20Strategies%20in%20MQL5%20(Part%2030)%3A%20Creating%20a%20Price%20Action%20AB-CD%20Harmonic%20Pattern%20with%20Visual%20Feedback%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909149539689007&fz_uniq=5049031951433704358&sv=2552)

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