---
title: Automating Trading Strategies in MQL5 (Part 27): Creating a Price Action Crab Harmonic Pattern with Visual Feedback
url: https://www.mql5.com/en/articles/19099
categories: Trading Systems, Expert Advisors
relevance_score: 10
scraped_at: 2026-01-22T17:21:28.638997
---

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/19099&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049072474450142309)

MetaTrader 5 / Trading systems


### Introduction

In our [previous article](https://www.mql5.com/en/articles/19077) (Part 26), we developed a [Pin Bar](https://www.mql5.com/go?link=https://priceaction.com/price-action-university/strategies/pin-bar/ "https://priceaction.com/price-action-university/strategies/pin-bar/") Averaging system in [MetaQuotes Language 5](https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5 "https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5") (MQL5) that utilized pin bar candlestick patterns to initiate trades and manage multiple positions through an averaging strategy, complete with a dynamic dashboard for real-time oversight. In Part 27, we create a [Crab Pattern system](https://www.mql5.com/go?link=https://howtotrade.com/chart-patterns/crab-pattern/ "https://howtotrade.com/chart-patterns/crab-pattern/") that identifies bullish and bearish Crab harmonic patterns using pivot points and Fibonacci ratios, automating trades with precise entry, stop loss, and take-profit levels, enhanced by visual chart objects like triangles and trendlines for clear pattern representation. We will cover the following topics:

1. [Understanding the Crab Harmonic Pattern Framework](https://www.mql5.com/en/articles/19099#para1)
2. [Implementation in MQL5](https://www.mql5.com/en/articles/19099#para2)
3. [Backtesting](https://www.mql5.com/en/articles/19099#para3)
4. [Conclusion](https://www.mql5.com/en/articles/19099#para4)

By the end, you’ll have a sophisticated MQL5 strategy for harmonic pattern trading, ready for customization—let’s dive in!

### Understanding the Crab Harmonic Pattern Framework

The [Crab pattern](https://www.mql5.com/go?link=https://howtotrade.com/chart-patterns/crab-pattern/ "https://howtotrade.com/chart-patterns/crab-pattern/") is a harmonic trading formation defined by five key swing points—X, A, B, C, and D—and exists in two forms: a bullish pattern and a bearish pattern. In a bullish Crab, the structure forms a low-high-low-high-low sequence where point X is a swing low, point A a swing high, point B a swing low (retracing 0.618 of XA), point C a swing high (extending 0.382 to 0.886 of AB), and point D a swing low (extending 1.618 of XA, positioned below X). Conversely, a bearish Crab forms a high-low-high-low-high sequence, with point X as a swing high, point A a swing low, point B a swing high, point C a swing low, and point D a swing high (extending 1.618 of XA, positioned above X). Below are the visualized pattern types.

Bullish Crab Harmonic Pattern:

![BULLISH CRAB HARMONIC PATTERN](https://c.mql5.com/2/162/Screenshot_2025-08-07_231130.png)

Bearish Crab Harmonic Pattern:

![BEARISH CRAB HARMONIC PATTERN](https://c.mql5.com/2/162/Screenshot_2025-08-07_231148.png)

To identify the patterns, below is our structured approach:

- Defining the XA Leg: The initial impulsive move from point X to point A establishes the pattern's foundation, determining the direction (downward for bullish, upward for bearish) and serving as the reference for Fibonacci calculations.
- Establishing the AB Leg: Point B should retrace approximately 0.618 of the XA leg, confirming a correction without reversing the initial move too aggressively.
- Analyzing the BC Leg: This leg should extend between 0.382 and 0.886 of the AB leg, creating a sharp counter-move that sets up the final extension.
- Setting the CD Leg: The final leg should extend 1.618 of the XA leg, marking the potential reversal zone at point D, where the pattern completes and a trade signal is generated.

By applying these [geometric and Fibonacci-based criteria](https://www.metatrader5.com/en/terminal/help/objects/fibo/fibo_retracement "https://www.metatrader5.com/en/terminal/help/objects/fibo/fibo_retracement"), our trading system will systematically detect valid Crab patterns in price data. Once identified, the system will visualize the formation on the chart with triangles, trend lines, labels for points X, A, B, C, and D, and dotted lines for entry and take-profit levels. This setup will enable automated execution of trades at the D point with calculated stop loss and multi-level take profits, leveraging the pattern’s high-probability reversal nature for effective market entries. Let’s proceed to the implementation!

### Implementation in MQL5

To create the program in MQL5, open the [MetaEditor](https://www.metatrader5.com/en/automated-trading/metaeditor "https://www.metatrader5.com/en/automated-trading/metaeditor"), go to the Navigator, locate the Indicators folder, click on the "New" tab, and follow the prompts to create the file. Once it is made, in the coding environment, we will need to declare some [global variables](https://www.mql5.com/en/docs/basis/variables/global) that we will use throughout the program.

```
//+------------------------------------------------------------------+
//|                                             Crab Pattern EA.mq5. |
//|                        Copyright 2025, Forex Algo-Trader, Allan. |
//|                                 "https://t.me/Forex_Algo_Trader" |
//+------------------------------------------------------------------+
#property copyright   "Forex Algo-Trader, Allan"
#property link        "https://t.me/Forex_Algo_Trader"
#property version     "1.00"
#property description "This EA trades based on Crab Strategy"
#property strict

#include <Trade\Trade.mqh>                         //--- Include Trade library for order management
CTrade obj_Trade;                                  //--- Instantiate trade object for executing orders

//--- Input parameters for user configuration
input int    PivotLeft = 5;                        // Number of bars to the left for pivot identification
input int    PivotRight = 5;                       // Number of bars to the right for pivot identification
input double Tolerance = 0.10;                     // Allowed deviation for Fibonacci levels (10% of XA move)
input double LotSize = 0.01;                       // Lot size for opening new trade positions
input bool   AllowTrading = true;                  // Enable or disable automated trading functionality

//---------------------------------------------------------------------------
//--- Crab pattern definition:
//--- Bullish Crab:
//--- Pivots (X-A-B-C-D): X swing low, A swing high, B swing low, C swing high, D swing low.
//--- Normally XA > 0; Ideal B = A - 0.5*(A-X); Legs within specified ranges.
//--- Bearish Crab:
//--- Pivots (X-A-B-C-D): X swing high, A swing low, B swing high, C swing low, D swing high.
//--- Normally XA > 0; Ideal B = A + 0.5*(X-A); Legs within specified ranges.
//---------------------------------------------------------------------------

struct Pivot {                                     //--- Define structure for pivot points
   datetime time;                                  //--- Store time of pivot bar
   double   price;                                 //--- Store price (high for swing high, low for swing low)
   bool     isHigh;                                //--- Indicate true for swing high, false for swing low
};

Pivot pivots[];                                    //--- Declare array to store pivot points
int      g_patternFormationBar = -1;               //--- Store bar index of pattern formation (-1 if none)
datetime g_lockedPatternX = 0;                     //--- Store X pivot time for locked pattern
```

We begin the implementation of the [Crab Pattern](https://www.mql5.com/go?link=https://howtotrade.com/chart-patterns/crab-pattern/ "https://howtotrade.com/chart-patterns/crab-pattern/") by including the "<Trade\\Trade.mqh>" library and instantiating "obj\_Trade" as a [CTrade](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade) object to facilitate order management, such as sending buy and sell requests. Then, we proceed to define [input parameters](https://www.mql5.com/en/docs/basis/variables/inputvariables) for user customization: "PivotLeft" and "PivotRight" at 5 bars each to specify the lookback for identifying swing pivots, "Tolerance" at 0.10 for Fibonacci deviation allowance, "LotSize" at 0.01 for trade volume, and "AllowTrading" as true to enable automated execution.

Next, we define the "Pivot" [structure](https://www.mql5.com/en/docs/basis/types/classes) with "time" (datetime), "price" ( [double](https://www.mql5.com/en/docs/basis/types/double)), and "isHigh" (bool) to store swing points, declare "pivots" as an array of "Pivot", and initialize globals "g\_patternFormationBar" as -1 to track pattern formation bars and "g\_lockedPatternX" as 0 to lock the X pivot time for confirmation, setting up the foundation for pattern identification. For visualization, we can have functions to draw lines, labels, and triangles.

```
//+------------------------------------------------------------------+
//| Draw filled triangle on chart                                    |
//+------------------------------------------------------------------+
void DrawTriangle(string name, datetime t1, double p1, datetime t2, double p2, datetime t3, double p3, color cl, int width, bool fill, bool back) {
   if (ObjectCreate(0, name, OBJ_TRIANGLE, 0, t1, p1, t2, p2, t3, p3)) { //--- Create triangle with three points
      ObjectSetInteger(0, name, OBJPROP_COLOR, cl); //--- Set triangle color
      ObjectSetInteger(0, name, OBJPROP_STYLE, STYLE_SOLID); //--- Set solid line style
      ObjectSetInteger(0, name, OBJPROP_WIDTH, width); //--- Set line width
      ObjectSetInteger(0, name, OBJPROP_FILL, fill); //--- Enable or disable fill
      ObjectSetInteger(0, name, OBJPROP_BACK, back); //--- Set background or foreground
   }
}

//+------------------------------------------------------------------+
//| Draw trend line on chart                                         |
//+------------------------------------------------------------------+
void DrawTrendLine(string name, datetime t1, double p1, datetime t2, double p2, color cl, int width, int style) {
   if (ObjectCreate(0, name, OBJ_TREND, 0, t1, p1, t2, p2)) { //--- Create trend line between two points
      ObjectSetInteger(0, name, OBJPROP_COLOR, cl); //--- Set line color
      ObjectSetInteger(0, name, OBJPROP_STYLE, style); //--- Set line style (solid, dotted, etc.)
      ObjectSetInteger(0, name, OBJPROP_WIDTH, width); //--- Set line width
   }
}

//+------------------------------------------------------------------+
//| Draw dotted horizontal line on chart                             |
//+------------------------------------------------------------------+
void DrawDottedLine(string name, datetime t1, double p, datetime t2, color lineColor) {
   if (ObjectCreate(0, name, OBJ_TREND, 0, t1, p, t2, p)) { //--- Create horizontal dotted line
      ObjectSetInteger(0, name, OBJPROP_COLOR, lineColor); //--- Set line color
      ObjectSetInteger(0, name, OBJPROP_STYLE, STYLE_DOT); //--- Set dotted style
      ObjectSetInteger(0, name, OBJPROP_WIDTH, 1); //--- Set line width to 1
   }
}

//+------------------------------------------------------------------+
//| Draw anchored text label for pivots                              |
//+------------------------------------------------------------------+
void DrawTextEx(string name, string text, datetime t, double p, color cl, int fontsize, bool isHigh) {
   if (ObjectCreate(0, name, OBJ_TEXT, 0, t, p)) { //--- Create text label at specified coordinates
      ObjectSetString(0, name, OBJPROP_TEXT, text); //--- Set label text content
      ObjectSetInteger(0, name, OBJPROP_COLOR, cl); //--- Set text color
      ObjectSetInteger(0, name, OBJPROP_FONTSIZE, fontsize); //--- Set font size
      ObjectSetString(0, name, OBJPROP_FONT, "Arial Bold"); //--- Set font to Arial Bold
      if (isHigh) {                                //--- Check if pivot is swing high
         ObjectSetInteger(0, name, OBJPROP_ANCHOR, ANCHOR_BOTTOM); //--- Anchor label above pivot
      } else {                                     //--- Handle swing low
         ObjectSetInteger(0, name, OBJPROP_ANCHOR, ANCHOR_TOP); //--- Anchor label below pivot
      }
      ObjectSetInteger(0, name, OBJPROP_ALIGN, ALIGN_CENTER); //--- Center-align text
   }
}
```

Here, we implement visualization functions for the program to draw chart objects that represent the Crab harmonic pattern and its trade levels. First, we create the "DrawTriangle" function, which uses [ObjectCreate](https://www.mql5.com/en/docs/objects/objectcreate) to draw a filled triangle ( [OBJ\_TRIANGLE](https://www.mql5.com/en/docs/constants/objectconstants/enum_object)) with three points defined by times ("t1", "t2", "t3") and prices ("p1", "p2", "p3"), setting [OBJPROP\_COLOR](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer) to the specified color, "OBJPROP\_STYLE" to "STYLE\_SOLID", "OBJPROP\_WIDTH" to the given width, "OBJPROP\_FILL" to enable or disable filling, and "OBJPROP\_BACK" to set background or foreground placement with the [ObjectSetInteger](https://www.mql5.com/en/docs/objects/objectsetinteger) function.

Then, we proceed to implement the "DrawTrendLine" function, which creates a trend line ("OBJ\_TREND") between two points using "ObjectCreate", configuring "OBJPROP\_COLOR", "OBJPROP\_STYLE" (solid, dotted, etc.), and [OBJPROP\_WIDTH](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer) with "ObjectSetInteger" for customizable line appearance. Next, we develop the "DrawDottedLine" function, which draws a horizontal dotted line ( [OBJ\_TREND](https://www.mql5.com/en/docs/constants/objectconstants/enum_object)) at a specified price from "t1" to "t2" and uses the same logic. Last, we implement the "DrawTextEx" function, which creates a text label ( [OBJ\_TEXT](https://www.mql5.com/en/docs/constants/objectconstants/enum_object)) at coordinates ("t", "p") with the object creation function and uses the same format as previous functions, ensuring a clear visual representation of the Crab pattern and trade levels on the chart. We can now proceed to the [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) event handler and try to find pivot points that we can use later on for pattern identification. Here is the logic we use to achieve that.

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick() {
   static datetime lastBarTime = 0;               //--- Store time of last processed bar
   datetime currentBarTime = iTime(_Symbol, _Period, 1); //--- Get time of current confirmed bar
   if (currentBarTime == lastBarTime) return;     //--- Exit if no new bar
   lastBarTime = currentBarTime;                  //--- Update last processed bar time
   ArrayResize(pivots, 0);                        //--- Clear pivot array for fresh analysis
   int barsCount = Bars(_Symbol, _Period);        //--- Retrieve total number of bars
   int start = PivotLeft;                         //--- Set starting index for pivot detection
   int end = barsCount - PivotRight;              //--- Set ending index for pivot detection
   for (int i = end - 1; i >= start; i--) {       //--- Iterate through bars to identify pivots
      bool isPivotHigh = true;                    //--- Assume bar is a swing high
      bool isPivotLow = true;                     //--- Assume bar is a swing low
      double currentHigh = iHigh(_Symbol, _Period, i); //--- Get current bar high price
      double currentLow = iLow(_Symbol, _Period, i); //--- Get current bar low price
      for (int j = i - PivotLeft; j <= i + PivotRight; j++) { //--- Check surrounding bars
         if (j < 0 || j >= barsCount) continue;   //--- Skip out-of-bounds indices
         if (j == i) continue;                    //--- Skip current bar
         if (iHigh(_Symbol, _Period, j) > currentHigh) isPivotHigh = false; //--- Invalidate swing high
         if (iLow(_Symbol, _Period, j) < currentLow) isPivotLow = false; //--- Invalidate swing low
      }
      if (isPivotHigh || isPivotLow) {            //--- Check if bar is a pivot
         Pivot p;                                 //--- Create new pivot structure
         p.time = iTime(_Symbol, _Period, i);     //--- Set pivot bar time
         p.price = isPivotHigh ? currentHigh : currentLow; //--- Set pivot price
         p.isHigh = isPivotHigh;                  //--- Set pivot type
         int size = ArraySize(pivots);            //--- Get current pivot array size
         ArrayResize(pivots, size + 1);           //--- Resize pivot array
         pivots[size] = p;                        //--- Add pivot to array
      }
   }
}
```

We proceed to implement the initial logic of the [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) event handler for our program to detect swing pivots, forming the basis for identifying Crab harmonic patterns. First, we check for a new bar by comparing "lastBarTime" (static, initialized to 0) with "currentBarTime" from [iTime](https://www.mql5.com/en/docs/series/itime) at shift 1 to avoid using the current incomplete bar for the current symbol and period, exiting if unchanged, and updating "lastBarTime" if a new bar is detected. Then, we proceed to clear the "pivots" array with [ArrayResize](https://www.mql5.com/en/docs/array/arraysize) to ensure a fresh analysis. Next, we retrieve the total bar count with [Bars](https://www.mql5.com/en/docs/series/bars), set the pivot detection range from "start" (equal to "PivotLeft") to "end" (total bars minus "PivotRight"), and iterate through bars from "end - 1" to "start".

For each bar, we assume it’s a swing high ("isPivotHigh" true) and low ("isPivotLow" true), get its high and low prices with [iHigh](https://www.mql5.com/en/docs/series/ihigh) and "iLow", and check surrounding bars within "PivotLeft" and "PivotRight" using "iHigh" and [iLow](https://www.mql5.com/en/docs/series/ilow) to invalidate the pivot if any neighboring bar has a higher high or lower low. Last, if the bar remains a valid pivot (high or low), we create a "Pivot" [structure](https://www.mql5.com/en/docs/basis/types/classes), set its "time" with "iTime", "price" to the high or low based on "isPivotHigh", and "isHigh" flag, then append it to the "pivots" array with "ArrayResize" and store it. When we print the array, we have the following outcome.

![PIVOTS DATA](https://c.mql5.com/2/162/Screenshot_2025-08-07_214134.png)

With the data, we can extract the pivot points, and if we have enough pivots, we can analyze and detect the patterns. Here is the logic we implement to achieve that.

```
int pivotCount = ArraySize(pivots);            //--- Get total number of pivots
if (pivotCount < 5) {                          //--- Check if insufficient pivots
   g_patternFormationBar = -1;                 //--- Reset pattern formation bar
   g_lockedPatternX = 0;                       //--- Reset locked X pivot
   return;                                     //--- Exit function
}
Pivot X = pivots[pivotCount - 5];              //--- Extract X pivot (earliest)
Pivot A = pivots[pivotCount - 4];              //--- Extract A pivot
Pivot B = pivots[pivotCount - 3];              //--- Extract B pivot
Pivot C = pivots[pivotCount - 2];              //--- Extract C pivot
Pivot D = pivots[pivotCount - 1];              //--- Extract D pivot (latest)
bool patternFound = false;                     //--- Initialize pattern detection flag
if (X.isHigh && !A.isHigh && B.isHigh && !C.isHigh && D.isHigh) { //--- Check bearish Crab pattern
   double diff = X.price - A.price;            //--- Calculate XA leg difference
   if (diff > 0) {                             //--- Ensure positive XA move
      double idealB = A.price + 0.618 * diff;  //--- Compute ideal B (0.618 retracement)
      if (MathAbs(B.price - idealB) <= Tolerance * diff) { //--- Verify B within tolerance
         double AB = B.price - A.price;        //--- Calculate AB leg length
         double BC = B.price - C.price;        //--- Calculate BC leg length
         if (BC >= 0.382 * AB && BC <= 0.886 * AB) { //--- Check BC within Fibonacci range
            double extension = D.price - A.price; //--- Calculate AD extension
            if (MathAbs(extension - 1.618 * diff) <= Tolerance * diff && D.price > X.price) { //--- Verify 1.618 extension and D > X
               patternFound = true;            //--- Confirm bearish pattern
            }
         }
      }
   }
}
if (!X.isHigh && A.isHigh && !B.isHigh && C.isHigh && !D.isHigh) { //--- Check bullish Crab pattern
   double diff = A.price - X.price;            //--- Calculate XA leg difference
   if (diff > 0) {                             //--- Ensure positive XA move
      double idealB = A.price - 0.618 * diff;  //--- Compute ideal B (0.618 retracement)
      if (MathAbs(B.price - idealB) <= Tolerance * diff) { //--- Verify B within tolerance
         double AB = A.price - B.price;        //--- Calculate AB leg length
         double BC = C.price - B.price;        //--- Calculate BC leg length
         if (BC >= 0.382 * AB && BC <= 0.886 * AB) { //--- Check BC within Fibonacci range
            double extension = A.price - D.price; //--- Calculate AD extension
            if (MathAbs(extension - 1.618 * diff) <= Tolerance * diff && D.price < X.price) { //--- Verify 1.618 extension and D < X
               patternFound = true;            //--- Confirm bullish pattern
            }
         }
      }
   }
}
```

To identify the patterns, we use Fibonacci-based criteria. First, we retrieve the total number of pivots with " [ArraySize(pivots)](https://www.mql5.com/en/docs/array/arraysize)" and store it in "pivotCount", exiting with "g\_patternFormationBar" and "g\_lockedPatternX" reset to -1 and 0, respectively if fewer than 5 pivots are found, as the Crab pattern requires X, A, B, C, and D points. Then, we proceed to extract the last five pivots from the "pivots" array, assigning "X" (earliest), "A", "B", "C", and "D" (latest) to represent the pattern’s structure.

Next, we check for a bearish Crab pattern by verifying the sequence (X high, A low, B high, C low, D high), calculating the XA leg difference ("X.price - A.price"), ensuring it’s positive, computing the ideal B point as "A.price + 0.618 \* diff", and confirming B is within "Tolerance \* diff" using [MathAbs](https://www.mql5.com/en/docs/math/mathabs); we then validate the BC leg (0.382 to 0.886 of AB) and the AD extension (1.618 of XA with D above X), setting "patternFound" to true if all conditions are met. Last, we check for a bullish Crab pattern (X low, A high, B low, C high, D low), calculating XA as "A.price - X.price", ensuring it’s positive, verifying B at 0.618 retracement, BC within 0.382 to 0.886 of AB, and AD at 1.618 of XA with D below X, setting "patternFound" to true if valid. If the pattern is found, we can proceed to visualize it on the chart.

```
string patternType = "";                                                //--- Initialize pattern type
if (patternFound) {                                                     //--- Check if pattern detected
   if (D.price > X.price) patternType = "Bearish";                      //--- Set bearish pattern (sell signal)
   else if (D.price < X.price) patternType = "Bullish";                 //--- Set bullish pattern (buy signal)
}
if (patternFound) {                                                     //--- Process valid Crab pattern
   Print(patternType, " Crab pattern detected at ", TimeToString(D.time, TIME_DATE|TIME_MINUTES|TIME_SECONDS)); //--- Log pattern detection
   string signalPrefix = "CR_" + IntegerToString(X.time);               //--- Generate unique prefix for objects
   color triangleColor = (patternType == "Bullish") ? clrBlue : clrRed; //--- Set triangle color based on pattern
   DrawTriangle(signalPrefix + "_Triangle1", X.time, X.price, A.time, A.price, B.time, B.price, triangleColor, 2, true, true); //--- Draw XAB triangle
   DrawTriangle(signalPrefix + "_Triangle2", B.time, B.price, C.time, C.price, D.time, D.price, triangleColor, 2, true, true); //--- Draw BCD triangle
}
```

To classify and visualize the detected patterns on the chart, we initialize "patternType" as an empty string to store whether the pattern is bullish or bearish. Then, if "patternFound" is true, we determine the pattern type by comparing "D.price" to "X.price": setting "patternType" to "Bearish" if D is above X (indicating a sell signal) or "Bullish" if D is below X (indicating a buy signal). Next, when a valid Crab pattern is confirmed, we log the detection with [Print](https://www.mql5.com/en/docs/common/print), outputting the "patternType" and the time of the D pivot using "TimeToString" formatted with date, minutes, and seconds.

Last, we create a unique identifier "signalPrefix" as "CR\_" concatenated with "X.time" converted to a string, set "triangleColor" to blue for bullish or red for bearish patterns, and call "DrawTriangle" twice to visualize the pattern: first for the XAB triangle (connecting X, A, B) and then for the BCD triangle (connecting B, C, D), using "signalPrefix" with suffixes "\_Triangle1" and "\_Triangle2", the respective pivot times and prices, "triangleColor", a width of 2, and enabling fill and background display, ensuring clear identification and visual representation of detected Crab patterns for trading decisions. Here is the milestone we have.

![PATTERN WITH TRIANGLES](https://c.mql5.com/2/162/Screenshot_2025-08-07_215837.png)

From the image, we can see that we can map and visualize the detected pattern correctly. We now need to continue mapping the trendlines to fully make it visible within boundaries and add a label to it for easier identification of the levels.

```
DrawTrendLine(signalPrefix + "_TL_XA", X.time, X.price, A.time, A.price, clrBlack, 2, STYLE_SOLID); //--- Draw XA trend line
DrawTrendLine(signalPrefix + "_TL_AB", A.time, A.price, B.time, B.price, clrBlack, 2, STYLE_SOLID); //--- Draw AB trend line
DrawTrendLine(signalPrefix + "_TL_BC", B.time, B.price, C.time, C.price, clrBlack, 2, STYLE_SOLID); //--- Draw BC trend line
DrawTrendLine(signalPrefix + "_TL_CD", C.time, C.price, D.time, D.price, clrBlack, 2, STYLE_SOLID); //--- Draw CD trend line
DrawTrendLine(signalPrefix + "_TL_XB", X.time, X.price, B.time, B.price, clrBlack, 2, STYLE_SOLID); //--- Draw XB trend line
DrawTrendLine(signalPrefix + "_TL_BD", B.time, B.price, D.time, D.price, clrBlack, 2, STYLE_SOLID); //--- Draw BD trend line
double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT); //--- Retrieve symbol point size
double offset = 15 * point;                    //--- Calculate label offset (15 points)
double textY_X = X.isHigh ? X.price + offset : X.price - offset; //--- Set X label Y coordinate
double textY_A = A.isHigh ? A.price + offset : A.price - offset; //--- Set A label Y coordinate
double textY_B = B.isHigh ? B.price + offset : B.price - offset; //--- Set B label Y coordinate
double textY_C = C.isHigh ? C.price + offset : C.price - offset; //--- Set C label Y coordinate
double textY_D = D.isHigh ? D.price + offset : D.price - offset; //--- Set D label Y coordinate
DrawTextEx(signalPrefix + "_Text_X", "X", X.time, textY_X, clrBlack, 11, X.isHigh); //--- Draw X pivot label
DrawTextEx(signalPrefix + "_Text_A", "A", A.time, textY_A, clrBlack, 11, A.isHigh); //--- Draw A pivot label
DrawTextEx(signalPrefix + "_Text_B", "B", B.time, textY_B, clrBlack, 11, B.isHigh); //--- Draw B pivot label
DrawTextEx(signalPrefix + "_Text_C", "C", C.time, textY_C, clrBlack, 11, C.isHigh); //--- Draw C pivot label
DrawTextEx(signalPrefix + "_Text_D", "D", D.time, textY_D, clrBlack, 11, D.isHigh); //--- Draw D pivot label
datetime centralTime = (X.time + B.time) / 2;                                       //--- Calculate central label time
double centralPrice = D.price;                                                      //--- Set central label price
if (ObjectCreate(0, signalPrefix + "_Text_Center", OBJ_TEXT, 0, centralTime, centralPrice)) { //--- Create central pattern label
   ObjectSetString(0, signalPrefix + "_Text_Center", OBJPROP_TEXT, patternType == "Bullish" ? "Bullish Crab" : "Bearish Crab"); //--- Set pattern name
   ObjectSetInteger(0, signalPrefix + "_Text_Center", OBJPROP_COLOR, clrBlack);     //--- Set text color
   ObjectSetInteger(0, signalPrefix + "_Text_Center", OBJPROP_FONTSIZE, 11);        //--- Set font size
   ObjectSetString(0, signalPrefix + "_Text_Center", OBJPROP_FONT, "Arial Bold");   //--- Set font type
   ObjectSetInteger(0, signalPrefix + "_Text_Center", OBJPROP_ALIGN, ALIGN_CENTER); //--- Center-align text
}
```

To depict the pattern structure, we continue to add lines and labels. First, we draw six trend lines using "DrawTrendLine" with the unique "signalPrefix" to connect key pivot points: XA, AB, BC, CD, XB, and BD, each with endpoints defined by their respective pivot times and prices (e.g., "X.time", "X.price"), using "clrBlack" for color, a width of 2, and [STYLE\_SOLID](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer) for a solid line style, outlining the XABCD structure and supporting legs. Then, we proceed to calculate a label offset by retrieving the symbol’s point size with " [SymbolInfoDouble(\_Symbol, SYMBOL\_POINT)](https://www.mql5.com/en/docs/marketinformation/symbolinfodouble)" and multiplying by 15, determining Y-coordinates for pivot labels ("textY\_X", "textY\_A", "textY\_B", "textY\_C", "textY\_D") by adding or subtracting the offset based on whether each pivot is a swing high ("isHigh" true) or low, ensuring labels appear above highs and below lows.

Next, we use "DrawTextEx" to create text labels for pivots X, A, B, C, and D, each with "signalPrefix" and suffixes like "\_Text\_X", displaying the respective letter, positioned at the pivot time and adjusted Y-coordinate, using "clrBlack", font size 11, and the pivot’s "isHigh" status for anchoring. Last, we calculate a central label position at "centralTime" as the midpoint of "X.time" and "B.time" and "centralPrice" at "D.price", creating a text object with [ObjectCreate](https://www.mql5.com/en/docs/objects/objectcreate) named "signalPrefix + '\_Text\_Center'", setting [OBJPROP\_TEXT](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_string) to "Bullish Crab" or "Bearish Crab" based on "patternType", and configuring "OBJPROP\_COLOR" to "clrBlack", "OBJPROP\_FONTSIZE" to 11, "OBJPROP\_FONT" to "Arial Bold", and "OBJPROP\_ALIGN" to "ALIGN\_CENTER" with [ObjectSetString](https://www.mql5.com/en/docs/objects/objectsetstring) and "ObjectSetInteger". This ensures a comprehensive visual representation of the Crab pattern’s structure and type on the chart. When we run the program, here is a visualization of what we get.

![CRAB PATTERN WITH EDGES AND LABELS](https://c.mql5.com/2/162/Screenshot_2025-08-07_220704.png)

From the image, we can see that we have added the edges and the labels to the pattern, making it more revealing and illustrative. What we need to do next is determine the trade levels for the pattern.

```
datetime lineStart = D.time;                                     //--- Set start time for trade level lines
datetime lineEnd = D.time + PeriodSeconds(_Period) * 2;          //--- Set end time for trade level lines
double entryPriceLevel, TP1Level, TP2Level, TP3Level, tradeDiff; //--- Declare trade level variables
if (patternType == "Bullish") {                                  //--- Handle bullish trade levels
   entryPriceLevel = SymbolInfoDouble(_Symbol, SYMBOL_ASK);      //--- Set entry at ask price
   TP3Level = C.price;                                           //--- Set TP3 at C pivot price
   tradeDiff = TP3Level - entryPriceLevel;                       //--- Calculate total trade distance
   TP1Level = entryPriceLevel + tradeDiff / 3;                   //--- Set TP1 at 1/3 of distance
   TP2Level = entryPriceLevel + 2 * tradeDiff / 3;               //--- Set TP2 at 2/3 of distance
} else {                                                         //--- Handle bearish trade levels
   entryPriceLevel = SymbolInfoDouble(_Symbol, SYMBOL_BID);      //--- Set entry at bid price
   TP3Level = C.price;                                           //--- Set TP3 at C pivot price
   tradeDiff = entryPriceLevel - TP3Level;                       //--- Calculate total trade distance
   TP1Level = entryPriceLevel - tradeDiff / 3;                   //--- Set TP1 at 1/3 of distance
   TP2Level = entryPriceLevel - 2 * tradeDiff / 3;               //--- Set TP2 at 2/3 of distance
}
DrawDottedLine(signalPrefix + "_EntryLine", lineStart, entryPriceLevel, lineEnd, clrMagenta);  //--- Draw entry level line
DrawDottedLine(signalPrefix + "_TP1Line", lineStart, TP1Level, lineEnd, clrForestGreen);       //--- Draw TP1 level line
DrawDottedLine(signalPrefix + "_TP2Line", lineStart, TP2Level, lineEnd, clrGreen);             //--- Draw TP2 level line
DrawDottedLine(signalPrefix + "_TP3Line", lineStart, TP3Level, lineEnd, clrDarkGreen);         //--- Draw TP3 level line
datetime labelTime = lineEnd + PeriodSeconds(_Period) / 2;                                     //--- Set time for trade level labels
string entryLabel = patternType == "Bullish" ? "BUY (" : "SELL (";                             //--- Start entry label text
entryLabel += DoubleToString(entryPriceLevel, _Digits) + ")";                                  //--- Append entry price
DrawTextEx(signalPrefix + "_EntryLabel", entryLabel, labelTime, entryPriceLevel, clrMagenta, 11, true); //--- Draw entry label
string tp1Label = "TP1 (" + DoubleToString(TP1Level, _Digits) + ")";                           //--- Create TP1 label text
DrawTextEx(signalPrefix + "_TP1Label", tp1Label, labelTime, TP1Level, clrForestGreen, 11, true); //--- Draw TP1 label
string tp2Label = "TP2 (" + DoubleToString(TP2Level, _Digits) + ")";                           //--- Create TP2 label text
DrawTextEx(signalPrefix + "_TP2Label", tp2Label, labelTime, TP2Level, clrGreen, 11, true);     //--- Draw TP2 label
string tp3Label = "TP3 (" + DoubleToString(TP3Level, _Digits) + ")";                           //--- Create TP3 label text
DrawTextEx(signalPrefix + "_TP3Label", tp3Label, labelTime, TP3Level, clrDarkGreen, 11, true); //--- Draw TP3 label
```

Here, we continue defining and visualizing trade levels for the detected pattern. First, we set "lineStart" to the D pivot’s time ("D.time") and "lineEnd" to two periods ahead using " [PeriodSeconds](https://www.mql5.com/en/docs/common/periodseconds)(\_Period) \* 2", and declare variables "entryPriceLevel", "TP1Level", "TP2Level", "TP3Level", and "tradeDiff" for trade calculations. Then, for a bullish pattern ("patternType == 'Bullish'"), we set "entryPriceLevel" to the current ask price with [SymbolInfoDouble](https://www.mql5.com/en/docs/marketinformation/symbolinfodouble), "TP3Level" to the C pivot’s price, calculate "tradeDiff" as "TP3Level - entryPriceLevel", and compute "TP1Level" and "TP2Level" as one-third and two-thirds of "tradeDiff" added to "entryPriceLevel"; for a bearish pattern, we use the bid price, set "TP3Level" to C’s price, calculate "tradeDiff" as "entryPriceLevel - TP3Level", and compute "TP1Level" and "TP2Level" by subtracting one-third and two-thirds of the trade difference.

Next, we draw four dotted horizontal lines using "DrawDottedLine": an entry line at "entryPriceLevel" in magenta, and take-profit lines at "TP1Level" (forest green), "TP2Level" (green), and "TP3Level" (dark green), spanning from "lineStart" to "lineEnd". Last, we set "labelTime" to "lineEnd" plus half a period, create label texts with prices formatted via [DoubleToString](https://www.mql5.com/en/docs/convert/doubletostring) (e.g., "BUY (price)" or "SELL (price)" for entry, "TP1 (price)", etc.), and use "DrawTextEx" to draw these labels at "labelTime" with corresponding colors, font size 11, and anchored above the price levels. Upon compilation, we have the following outcome.

Bearish pattern:

![BEARISH](https://c.mql5.com/2/162/Screenshot_2025-08-07_223406.png)

Bullish pattern:

![BULLISH](https://c.mql5.com/2/162/Screenshot_2025-08-07_223459.png)

From the images, we can see that we have correctly mapped the trade levels. What we need to do now is initiate the actual trade positions, and that is all.

```
int currentBarIndex = Bars(_Symbol, _Period) - 1;       //--- Retrieve current bar index
if (g_patternFormationBar == -1) {                      //--- Check if no pattern is locked
   g_patternFormationBar = currentBarIndex;             //--- Lock current bar as formation bar
   g_lockedPatternX = X.time;                           //--- Lock X pivot time
   Print("Pattern detected on bar ", currentBarIndex, ". Waiting for confirmation on next bar."); //--- Log detection
   return;                                              //--- Exit function
}
if (currentBarIndex == g_patternFormationBar) {         //--- Check if still on formation bar
   Print("Pattern is repainting; still on locked formation bar ", currentBarIndex, ". No trade yet."); //--- Log repainting
   return;                                              //--- Exit function
}
if (currentBarIndex > g_patternFormationBar) {          //--- Check if new bar after formation
   if (g_lockedPatternX == X.time) {                    //--- Verify same X pivot for confirmation
      Print("Confirmed pattern (locked on bar ", g_patternFormationBar, "). Opening trade on bar ", currentBarIndex, "."); //--- Log confirmed pattern
      g_patternFormationBar = currentBarIndex;          //--- Update formation bar to current
      if (AllowTrading && !PositionSelect(_Symbol)) {   //--- Check trading allowed and no open position
         double entryPriceTrade = 0, stopLoss = 0, takeProfit = 0; //--- Declare trade parameters
         point = SymbolInfoDouble(_Symbol, SYMBOL_POINT); //--- Update point value
         bool tradeResult = false;                      //--- Initialize trade result flag
         if (patternType == "Bullish") {                //--- Process bullish trade
            entryPriceTrade = SymbolInfoDouble(_Symbol, SYMBOL_ASK); //--- Set entry at ask price
            double diffTrade = TP2Level - entryPriceTrade; //--- Calculate trade distance
            stopLoss = entryPriceTrade - diffTrade * 3; //--- Set stop loss (3x distance)
            takeProfit = TP2Level;                      //--- Set take profit at TP2
            tradeResult = obj_Trade.Buy(LotSize, _Symbol, entryPriceTrade, stopLoss, takeProfit, "Crab Signal"); //--- Execute buy trade
            if (tradeResult) {                          //--- Check trade success
               Print("Buy order opened successfully."); //--- Log successful buy
            } else {                                    //--- Handle trade failure
               Print("Buy order failed: ", obj_Trade.ResultRetcodeDescription()); //--- Log failure reason
            }
         } else if (patternType == "Bearish") {         //--- Process bearish trade
            entryPriceTrade = SymbolInfoDouble(_Symbol, SYMBOL_BID); //--- Set entry at bid price
            double diffTrade = entryPriceTrade - TP2Level; //--- Calculate trade distance
            stopLoss = entryPriceTrade + diffTrade * 3; //--- Set stop loss (3x distance)
            takeProfit = TP2Level;                      //--- Set take profit at TP2
            tradeResult = obj_Trade.Sell(LotSize, _Symbol, entryPriceTrade, stopLoss, takeProfit, "Crab Signal"); //--- Execute sell trade
            if (tradeResult) {                          //--- Check trade success
               Print("Sell order opened successfully."); //--- Log successful sell
            } else {                                    //--- Handle trade failure
               Print("Sell order failed: ", obj_Trade.ResultRetcodeDescription()); //--- Log failure reason
            }
         }
      } else {                                          //--- Trading not allowed or position exists
         Print("A position is already open for ", _Symbol, ". No new trade executed."); //--- Log no trade
      }
   } else {                                            //--- Pattern has changed
      g_patternFormationBar = currentBarIndex;         //--- Update formation bar
      g_lockedPatternX = X.time;                       //--- Update locked X pivot
      Print("Pattern has changed; updating lock on bar ", currentBarIndex, ". Waiting for confirmation."); //--- Log pattern change
      return;                                          //--- Exit function
   }
}
} else {                                               //--- No valid pattern detected
   g_patternFormationBar = -1;                         //--- Reset formation bar
   g_lockedPatternX = 0;                               //--- Reset locked X pivot
}
```

First, we retrieve the current bar index with " [Bars(\_Symbol, \_Period)](https://www.mql5.com/en/docs/series/bars) \- 1" and store it in "currentBarIndex". Then, if no pattern is locked ("g\_patternFormationBar == -1"), we set "g\_patternFormationBar" to "currentBarIndex", lock the X pivot time in "g\_lockedPatternX" with "X.time", log the detection indicating a wait for confirmation, and exit. Next, if still on the formation bar ("currentBarIndex == g\_patternFormationBar"), we log that the pattern is repainting and exit to avoid premature trading.

Last, if a new bar has formed ("currentBarIndex > g\_patternFormationBar") and the X pivot matches "g\_lockedPatternX", we confirm the pattern, log it, update "g\_patternFormationBar", and check if trading is allowed with "AllowTrading" and no open positions exist via [PositionSelect](https://www.mql5.com/en/docs/trading/positionselect); for a bullish pattern, we set "entryPriceTrade" to the ask price, calculate "diffTrade" as "TP2Level - entryPriceTrade", set "stopLoss" three times this distance below, set "takeProfit" to "TP2Level", and execute a buy with "obj\_Trade.Buy" using "LotSize" and "Crab Signal" comment, logging success or failure; for a bearish pattern, we use the bid price, set "stopLoss" three times above, and execute a sell with "obj\_Trade.Sell"; if trading is disallowed or a position exists, we log no trade; if the pattern changes, we update the lock and wait; if no pattern is found, we reset "g\_patternFormationBar" and "g\_lockedPatternX", ensuring confirmed Crab patterns trigger trades with precise risk management. Lastly, we just need to delete the patterns from the chart when we remove the program.

```
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
   ObjectsDeleteAll(0, "CR_");                    //--- Remove all chart objects with "CR_" prefix
   ArrayResize(pivots, 0);                        //--- Clear pivots array
   g_patternFormationBar = -1;                    //--- Reset pattern formation bar index
   g_lockedPatternX = 0;                          //--- Reset locked pattern X pivot time
   ChartRedraw(0);                                //--- Redraw chart to reflect changes
}
```

Here, we implement the [OnDeinit](https://www.mql5.com/en/docs/event_handlers/ondeinit) event handler to ensure proper cleanup when the EA is removed from the chart. First, we remove all chart objects with the "CR\_" prefix using [ObjectsDeleteAll](https://www.mql5.com/en/docs/objects/ObjectDeleteAll) to clear visual elements like triangles, trendlines, and labels associated with Crab patterns. Then, we proceed to resize the "pivots" array to 0 with [ArrayResize](https://www.mql5.com/en/docs/array/arrayresize) to clear stored pivot data. Next, we reset "g\_patternFormationBar" to -1 and "g\_lockedPatternX" to 0 to clear pattern tracking variables. Last, we call [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw) to refresh the chart, ensuring it reflects the removal of all objects and data. This ensures a clean exit, freeing resources and preventing residual elements. Upon compilation, we have the following outcome.

Bearish signal:

![BEARISH SIGNAL](https://c.mql5.com/2/162/Screenshot_2025-08-07_225039.png)

Bullish signal:

![BULLISH SIGNAL](https://c.mql5.com/2/162/Screenshot_2025-08-07_225516.png)

From the image, we can see that we plot the harmonic pattern and are still able to trade it accordingly once it is confirmed, hence achieving our objective of identifying, plotting, and trading the pattern. The thing that remains is backtesting the program, and that is handled in the next section.

### Backtesting

After thorough backtesting, we have the following results.

Backtest graph:

![GRAPH](https://c.mql5.com/2/162/Screenshot_2025-08-08_010952.png)

Backtest report:

![REPORT](https://c.mql5.com/2/162/Screenshot_2025-08-08_011019.png)

### Conclusion

In conclusion, we’ve developed a [Crab Pattern](https://www.mql5.com/go?link=https://howtotrade.com/chart-patterns/crab-pattern/ "https://howtotrade.com/chart-patterns/crab-pattern/") system in MQL5, leveraging price action to identify bullish and bearish Crab harmonic patterns with precise [Fibonacci ratios](https://en.wikipedia.org/wiki/Fibonacci_sequence "https://en.wikipedia.org/wiki/Fibonacci_sequence"), automating trades with calculated entry, stop loss, and multi-level take-profit points, visualized through dynamic chart objects like triangles and trendlines.

Disclaimer: This article is for educational purposes only. Trading carries significant financial risks, and market volatility may result in losses. Thorough backtesting and careful risk management are crucial before deploying this program in live markets.

By leveraging the concepts and implementation presented, you can adapt this Crab pattern system to your trading style, enhancing your algorithmic strategies. Happy trading!

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/19099.zip "Download all attachments in the single ZIP archive")

[Crab\_Pattern\_EA.mq5](https://www.mql5.com/en/articles/download/19099/crab_pattern_ea.mq5 "Download Crab_Pattern_EA.mq5")(47.93 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/493724)**

![From Basic to Intermediate: Template and Typename (III)](https://c.mql5.com/2/112/Do_bdsico_ao_intermedirrio__Template_e_Typename_I___LOGO.png)[From Basic to Intermediate: Template and Typename (III)](https://www.mql5.com/en/articles/15669)

In this article, we will discuss the first part of the topic, which is not so easy for beginners to understand. In order not to get even more confused and to explain this topic correctly, we will divide the explanation into stages. We will devote this article to the first stage. However, although at the end of the article it may seem that we have reached the deadlock, in fact we will take a step towards another situation, which will be better understood in the next article.

![From Basic to Intermediate: Template and Typename (II)](https://c.mql5.com/2/110/Do_btsico_ao_intermedi0rio__Template_e_Typename_I___LOGO.png)[From Basic to Intermediate: Template and Typename (II)](https://www.mql5.com/en/articles/15668)

This article explains how to deal with one of the most difficult programming situations you can encounter: using different types in the same function or procedure template. Although we have spent most of our time focusing only on functions, everything covered here is useful and can be applied to procedures.

![CRUD Operations in Firebase using MQL](https://c.mql5.com/2/164/17854-crud-operations-in-firebase-logo__1.png)[CRUD Operations in Firebase using MQL](https://www.mql5.com/en/articles/17854)

This article offers a step-by-step guide to mastering CRUD (Create, Read, Update, Delete) operations in Firebase, focusing on its Realtime Database and Firestore. Discover how to use Firebase SDK methods to efficiently manage data in web and mobile apps, from adding new records to querying, modifying, and deleting entries. Explore practical code examples and best practices for structuring and handling data in real-time, empowering developers to build dynamic, scalable applications with Firebase’s flexible NoSQL architecture.

![Self Optimizing Expert Advisors in MQL5 (Part 12): Building Linear Classifiers Using Matrix Factorization](https://c.mql5.com/2/163/18987-self-optimizing-expert-advisors-logo.png)[Self Optimizing Expert Advisors in MQL5 (Part 12): Building Linear Classifiers Using Matrix Factorization](https://www.mql5.com/en/articles/18987)

This article explores the powerful role of matrix factorization in algorithmic trading, specifically within MQL5 applications. From regression models to multi-target classifiers, we walk through practical examples that demonstrate how easily these techniques can be integrated using built-in MQL5 functions. Whether you're predicting price direction or modeling indicator behavior, this guide lays a strong foundation for building intelligent trading systems using matrix methods.

[![](https://www.mql5.com/ff/si/d9hnbkyp2d47h07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fsignals%2Fmt5%2Fpage1%3Fpreset%3D2%26utm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dmax.profit.signals%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=hgyovyikvykcdukcncnktswvlctghemf&s=545653d14172edfb3c9c02ca8e948778c29f9c1b70be9a587e8d4b040fb23539&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=iqjhvtvjpuozkqlrtxneewhekcmvhbjy&ssn=1769091686600580919&ssn_dr=0&ssn_sr=0&fv_date=1769091686&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F19099&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Automating%20Trading%20Strategies%20in%20MQL5%20(Part%2027)%3A%20Creating%20a%20Price%20Action%20Crab%20Harmonic%20Pattern%20with%20Visual%20Feedback%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909168674544660&fz_uniq=5049072474450142309&sv=2552)

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